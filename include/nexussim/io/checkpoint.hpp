#pragma once

/**
 * @file checkpoint.hpp
 * @brief Binary checkpoint/restart system for simulation state serialization
 *
 * Features:
 * - Binary format with versioned header for forward compatibility
 * - Full state serialization: mesh, fields, material states, solver parameters
 * - Portable endianness handling
 * - Incremental checkpoint (save only changed fields)
 * - Round-trip verified: write → read → verify identical state
 *
 * File format:
 *   [Header: 64 bytes]
 *     magic (8 bytes): "NXSCKPT\0"
 *     version (4 bytes): format version (1)
 *     endian_check (4 bytes): 0x01020304 for endianness detection
 *     precision (4 bytes): sizeof(Real) — 4 or 8
 *     num_nodes (8 bytes): number of mesh nodes
 *     num_elements (8 bytes): total element count
 *     num_fields (4 bytes): number of state fields
 *     num_material_states (8 bytes): number of material integration points
 *     time (8 bytes): simulation time
 *     step (8 bytes): time step number
 *   [Mesh Coordinates Section]
 *     3 * num_nodes * sizeof(Real) bytes
 *   [Field Sections] × num_fields
 *     name_length (4 bytes)
 *     name (name_length bytes)
 *     field_type (4 bytes)
 *     field_location (4 bytes)
 *     num_entities (8 bytes)
 *     num_components (4 bytes)
 *     data (num_entities * num_components * sizeof(Real) bytes)
 *   [Material State Section]
 *     num_material_states × sizeof(MaterialStateData) bytes
 *
 * Reference: Inspired by OpenRadioss restart file format
 */

#include <nexussim/core/types.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <nexussim/physics/material.hpp>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

namespace nxs {
namespace io {

// ============================================================================
// Checkpoint File Header
// ============================================================================

struct CheckpointHeader {
    // Group 8-byte aligned fields together, then 4-byte fields, to avoid padding
    char magic[8];               // "NXSCKPT\0"                offset 0
    uint64_t num_nodes;          //                            offset 8
    uint64_t num_elements;       //                            offset 16
    uint64_t num_material_states;//                            offset 24
    double time;                 //                            offset 32
    int64_t step;                //                            offset 40
    uint32_t version;            // Format version             offset 48
    uint32_t endian_check;       // 0x01020304                 offset 52
    uint32_t precision;          // sizeof(Real)               offset 56
    uint32_t num_fields;         //                            offset 60

    CheckpointHeader()
        : num_nodes(0), num_elements(0)
        , num_material_states(0)
        , time(0.0), step(0)
        , version(1)
        , endian_check(0x01020304)
        , precision(sizeof(Real))
        , num_fields(0) {
        std::memcpy(magic, "NXSCKPT", 8);
    }

    bool is_valid() const {
        return std::memcmp(magic, "NXSCKPT", 7) == 0
            && version >= 1
            && endian_check == 0x01020304
            && precision == sizeof(Real);
    }
};

static_assert(sizeof(CheckpointHeader) == 64, "Header must be 64 bytes");

// ============================================================================
// Serialized Material State (fixed-size for binary I/O)
// ============================================================================

struct MaterialStateData {
    Real strain[6];
    Real stress[6];
    Real history[32];
    Real vol_strain;
    Real plastic_strain;
    Real temperature;
    Real damage;
    Real effective_strain_rate;
    Real dt;

    MaterialStateData() {
        std::memset(this, 0, sizeof(MaterialStateData));
        temperature = 293.15;
    }

    void from_material_state(const physics::MaterialState& ms) {
        std::memcpy(strain, ms.strain, 6 * sizeof(Real));
        std::memcpy(stress, ms.stress, 6 * sizeof(Real));
        std::memcpy(history, ms.history, 32 * sizeof(Real));
        vol_strain = ms.vol_strain;
        plastic_strain = ms.plastic_strain;
        temperature = ms.temperature;
        damage = ms.damage;
        effective_strain_rate = ms.effective_strain_rate;
        dt = ms.dt;
    }

    void to_material_state(physics::MaterialState& ms) const {
        std::memcpy(ms.strain, strain, 6 * sizeof(Real));
        std::memcpy(ms.stress, stress, 6 * sizeof(Real));
        std::memcpy(ms.history, history, 32 * sizeof(Real));
        ms.vol_strain = vol_strain;
        ms.plastic_strain = plastic_strain;
        ms.temperature = temperature;
        ms.damage = damage;
        ms.effective_strain_rate = effective_strain_rate;
        ms.dt = dt;
    }
};

// ============================================================================
// Checkpoint Writer
// ============================================================================

class CheckpointWriter {
public:
    explicit CheckpointWriter(const std::string& filename)
        : filename_(filename) {}

    /**
     * @brief Write full simulation state to checkpoint file
     * @param mesh The computational mesh
     * @param state The simulation state (fields)
     * @param material_states Optional material integration point data
     * @return true on success
     */
    bool write(const Mesh& mesh,
               const State& state,
               const std::vector<physics::MaterialState>& material_states = {}) {

        std::ofstream file(filename_, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Checkpoint: Cannot open file for writing: " << filename_ << "\n";
            return false;
        }

        // Build header
        CheckpointHeader header;
        header.num_nodes = mesh.num_nodes();
        header.num_elements = mesh.num_elements();
        header.num_fields = static_cast<uint32_t>(state.field_names().size());
        header.num_material_states = material_states.size();
        header.time = state.time();
        header.step = state.step();

        // Write header
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write mesh coordinates (3 * num_nodes Reals)
        write_mesh_coordinates(file, mesh);

        // Write state fields
        auto names = state.field_names();
        for (const auto& name : names) {
            write_field(file, name, state.field(name));
        }

        // Write material states
        if (!material_states.empty()) {
            write_material_states(file, material_states);
        }

        file.flush();
        bytes_written_ = file.tellp();
        return file.good();
    }

    /**
     * @brief Write only specified fields (incremental checkpoint)
     */
    bool write_fields_only(const State& state,
                            const std::vector<std::string>& field_names) {
        std::ofstream file(filename_, std::ios::binary);
        if (!file.is_open()) return false;

        // Simplified header for field-only checkpoint
        CheckpointHeader header;
        header.num_nodes = 0;  // Indicates no mesh data
        header.num_fields = static_cast<uint32_t>(field_names.size());
        header.time = state.time();
        header.step = state.step();

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto& name : field_names) {
            if (state.has_field(name)) {
                write_field(file, name, state.field(name));
            }
        }

        file.flush();
        bytes_written_ = file.tellp();
        return file.good();
    }

    std::size_t bytes_written() const { return bytes_written_; }

private:
    void write_mesh_coordinates(std::ofstream& file, const Mesh& mesh) {
        std::size_t n = mesh.num_nodes();
        const auto& coords = mesh.coordinates();

        // Write x, y, z arrays (SOA layout, same as Field storage)
        for (std::size_t comp = 0; comp < 3; ++comp) {
            auto span = coords.component(comp);
            file.write(reinterpret_cast<const char*>(span.data()),
                       static_cast<std::streamsize>(n * sizeof(Real)));
        }
    }

    void write_field(std::ofstream& file, const std::string& name,
                     const Field<Real>& field) {
        // Write field name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);

        // Write field metadata
        uint32_t ftype = static_cast<uint32_t>(field.type());
        uint32_t floc = static_cast<uint32_t>(field.location());
        uint64_t nentities = field.num_entities();
        uint32_t ncomp = static_cast<uint32_t>(field.num_components());

        file.write(reinterpret_cast<const char*>(&ftype), sizeof(ftype));
        file.write(reinterpret_cast<const char*>(&floc), sizeof(floc));
        file.write(reinterpret_cast<const char*>(&nentities), sizeof(nentities));
        file.write(reinterpret_cast<const char*>(&ncomp), sizeof(ncomp));

        // Write field data
        file.write(reinterpret_cast<const char*>(field.data()),
                   static_cast<std::streamsize>(nentities * ncomp * sizeof(Real)));
    }

    void write_material_states(std::ofstream& file,
                                const std::vector<physics::MaterialState>& states) {
        std::vector<MaterialStateData> data(states.size());
        for (std::size_t i = 0; i < states.size(); ++i) {
            data[i].from_material_state(states[i]);
        }
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(MaterialStateData)));
    }

    std::string filename_;
    std::size_t bytes_written_ = 0;
};

// ============================================================================
// Checkpoint Reader
// ============================================================================

class CheckpointReader {
public:
    explicit CheckpointReader(const std::string& filename)
        : filename_(filename) {}

    /**
     * @brief Read checkpoint header (without loading data)
     * @return true if header is valid
     */
    bool read_header() {
        std::ifstream file(filename_, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Checkpoint: Cannot open file for reading: " << filename_ << "\n";
            return false;
        }

        file.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!file.good() || !header_.is_valid()) {
            std::cerr << "Checkpoint: Invalid file format or incompatible precision\n";
            return false;
        }

        header_valid_ = true;
        return true;
    }

    /**
     * @brief Read full simulation state from checkpoint file
     * @param mesh The mesh to populate coordinates into
     * @param state The state to populate fields into
     * @param material_states Optional output for material states
     * @return true on success
     */
    bool read(Mesh& mesh, State& state,
              std::vector<physics::MaterialState>* material_states = nullptr) {

        std::ifstream file(filename_, std::ios::binary);
        if (!file.is_open()) return false;

        // Read header
        file.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!file.good() || !header_.is_valid()) return false;
        header_valid_ = true;

        // Set state time/step
        state.set_time(header_.time);
        state.set_step(header_.step);

        // Read mesh coordinates if present
        if (header_.num_nodes > 0) {
            read_mesh_coordinates(file, mesh);
        }

        // Read fields
        for (uint32_t i = 0; i < header_.num_fields; ++i) {
            read_field(file, state);
        }

        // Read material states
        if (material_states && header_.num_material_states > 0) {
            read_material_states(file, *material_states);
        }

        return file.good();
    }

    /**
     * @brief Read only specified fields from checkpoint
     */
    bool read_fields(State& state, const std::vector<std::string>& wanted_fields = {}) {
        std::ifstream file(filename_, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!file.good() || !header_.is_valid()) return false;

        state.set_time(header_.time);
        state.set_step(header_.step);

        // Skip mesh coordinates
        if (header_.num_nodes > 0) {
            file.seekg(static_cast<std::streamoff>(3 * header_.num_nodes * sizeof(Real)),
                       std::ios::cur);
        }

        // Read fields
        for (uint32_t i = 0; i < header_.num_fields; ++i) {
            read_field(file, state, wanted_fields);
        }

        return file.good();
    }

    // Header accessors
    const CheckpointHeader& header() const { return header_; }
    bool header_valid() const { return header_valid_; }
    Real time() const { return header_.time; }
    int64_t step() const { return header_.step; }
    uint64_t num_nodes() const { return header_.num_nodes; }
    uint64_t num_elements() const { return header_.num_elements; }

private:
    void read_mesh_coordinates(std::ifstream& file, Mesh& mesh) {
        std::size_t n = header_.num_nodes;
        auto& coords = mesh.coordinates();

        for (std::size_t comp = 0; comp < 3; ++comp) {
            auto span = coords.component(comp);
            file.read(reinterpret_cast<char*>(span.data()),
                      static_cast<std::streamsize>(n * sizeof(Real)));
        }
    }

    void read_field(std::ifstream& file, State& state,
                    const std::vector<std::string>& wanted = {}) {
        // Read name
        uint32_t name_len = 0;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read metadata
        uint32_t ftype, floc, ncomp;
        uint64_t nentities;
        file.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
        file.read(reinterpret_cast<char*>(&floc), sizeof(floc));
        file.read(reinterpret_cast<char*>(&nentities), sizeof(nentities));
        file.read(reinterpret_cast<char*>(&ncomp), sizeof(ncomp));

        std::size_t data_bytes = nentities * ncomp * sizeof(Real);

        // Check if we want this field
        bool want = wanted.empty();
        if (!want) {
            for (const auto& w : wanted) {
                if (w == name) { want = true; break; }
            }
        }

        if (!want) {
            // Skip field data
            file.seekg(static_cast<std::streamoff>(data_bytes), std::ios::cur);
            return;
        }

        // Create or update field in state
        if (state.has_field(name)) {
            auto& existing = state.field(name);
            file.read(reinterpret_cast<char*>(existing.data()),
                      static_cast<std::streamsize>(data_bytes));
        } else {
            // Create new field
            Field<Real> new_field(name,
                                   static_cast<FieldType>(ftype),
                                   static_cast<FieldLocation>(floc),
                                   nentities, ncomp);
            file.read(reinterpret_cast<char*>(new_field.data()),
                      static_cast<std::streamsize>(data_bytes));
            state.add_field(name, std::move(new_field));
        }
    }

    void read_material_states(std::ifstream& file,
                               std::vector<physics::MaterialState>& states) {
        std::size_t n = header_.num_material_states;
        std::vector<MaterialStateData> data(n);
        file.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(n * sizeof(MaterialStateData)));

        states.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            data[i].to_material_state(states[i]);
        }
    }

    std::string filename_;
    CheckpointHeader header_;
    bool header_valid_ = false;
};

// ============================================================================
// Checkpoint Manager (automatic periodic checkpointing)
// ============================================================================

class CheckpointManager {
public:
    CheckpointManager() = default;

    /**
     * @brief Configure checkpoint manager
     * @param base_path Base path for checkpoint files (e.g., "output/checkpoint")
     * @param interval_steps Write checkpoint every N steps (0 = disabled)
     * @param max_keep Maximum number of checkpoint files to keep (0 = unlimited)
     */
    void configure(const std::string& base_path,
                   int64_t interval_steps = 0,
                   int max_keep = 3) {
        base_path_ = base_path;
        interval_steps_ = interval_steps;
        max_keep_ = max_keep;
    }

    /**
     * @brief Check if checkpoint should be written at this step
     */
    bool should_write(int64_t step) const {
        if (interval_steps_ <= 0) return false;
        return (step % interval_steps_) == 0;
    }

    /**
     * @brief Write checkpoint if interval reached
     * @return true if checkpoint was written (or not needed), false on error
     */
    bool maybe_write(int64_t step, const Mesh& mesh, const State& state,
                     const std::vector<physics::MaterialState>& mat_states = {}) {
        if (!should_write(step)) return true;
        return write_checkpoint(step, mesh, state, mat_states);
    }

    /**
     * @brief Force write a checkpoint now
     */
    bool write_checkpoint(int64_t step, const Mesh& mesh, const State& state,
                           const std::vector<physics::MaterialState>& mat_states = {}) {
        std::string filename = make_filename(step);

        CheckpointWriter writer(filename);
        bool ok = writer.write(mesh, state, mat_states);

        if (ok) {
            checkpoint_files_.push_back(filename);
            total_checkpoints_++;
            total_bytes_ += writer.bytes_written();

            // Cleanup old checkpoints
            cleanup_old_checkpoints();
        }

        return ok;
    }

    /**
     * @brief Find the latest checkpoint file
     * @return filename, or empty string if none found
     */
    std::string find_latest() const {
        if (checkpoint_files_.empty()) return "";
        return checkpoint_files_.back();
    }

    /**
     * @brief Read the latest checkpoint
     */
    bool read_latest(Mesh& mesh, State& state,
                     std::vector<physics::MaterialState>* mat_states = nullptr) {
        std::string latest = find_latest();
        if (latest.empty()) return false;

        CheckpointReader reader(latest);
        return reader.read(mesh, state, mat_states);
    }

    // Statistics
    int total_checkpoints() const { return total_checkpoints_; }
    std::size_t total_bytes() const { return total_bytes_; }
    const std::vector<std::string>& checkpoint_files() const { return checkpoint_files_; }

    void print_summary() const {
        std::cout << "Checkpoint Manager: " << total_checkpoints_ << " checkpoints written"
                  << " (" << total_bytes_ / 1024 << " KB total)\n";
        if (!checkpoint_files_.empty()) {
            std::cout << "  Latest: " << checkpoint_files_.back() << "\n";
        }
    }

private:
    std::string make_filename(int64_t step) const {
        // Format: base_path_NNNNNNNN.nxs
        char buf[32];
        std::snprintf(buf, sizeof(buf), "_%08ld.nxs", static_cast<long>(step));
        return base_path_ + buf;
    }

    void cleanup_old_checkpoints() {
        if (max_keep_ <= 0) return;

        while (static_cast<int>(checkpoint_files_.size()) > max_keep_) {
            const std::string& old_file = checkpoint_files_.front();
            std::remove(old_file.c_str());
            checkpoint_files_.erase(checkpoint_files_.begin());
        }
    }

    std::string base_path_;
    int64_t interval_steps_ = 0;
    int max_keep_ = 3;

    std::vector<std::string> checkpoint_files_;
    int total_checkpoints_ = 0;
    std::size_t total_bytes_ = 0;
};

} // namespace io
} // namespace nxs
