#pragma once

/**
 * @file vtk_writer.hpp
 * @brief VTK format writer for visualization in ParaView
 *
 * Supports:
 * - Unstructured grid (.vtu) format
 * - Nodal and element fields
 * - Time series output
 * - Multiple element types
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <nexussim/physics/element.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <map>

namespace nxs {
namespace io {

/**
 * @brief VTK writer for unstructured grids
 *
 * Writes mesh and solution data in VTK XML format (.vtu)
 * Compatible with ParaView and VisIt
 */
class VTKWriter {
public:
    /**
     * @brief Constructor
     * @param filename Base filename (without .vtu extension)
     */
    VTKWriter(const std::string& filename);

    /**
     * @brief Write mesh and fields to VTK file
     * @param mesh Computational mesh
     * @param state Solution state (optional)
     * @param time Current simulation time (optional)
     */
    void write(const Mesh& mesh,
               const State* state = nullptr,
               Real time = 0.0);

    /**
     * @brief Write time series (creates .pvd collection file)
     * @param mesh Computational mesh
     * @param state Solution state
     * @param time Current time
     * @param step Time step number
     */
    void write_time_step(const Mesh& mesh,
                        const State& state,
                        Real time,
                        int step);

    /**
     * @brief Finalize time series (write .pvd file)
     */
    void finalize_time_series();

    /**
     * @brief Set output directory
     */
    void set_output_directory(const std::string& dir) {
        output_dir_ = dir;
    }

    /**
     * @brief Enable/disable ASCII output (default: binary)
     */
    void set_ascii_mode(bool ascii) {
        ascii_mode_ = ascii;
    }

private:
    /**
     * @brief Write VTK header
     */
    void write_header(std::ofstream& file);

    /**
     * @brief Write mesh geometry (points)
     */
    void write_points(std::ofstream& file, const Mesh& mesh);

    /**
     * @brief Write mesh topology (cells)
     */
    void write_cells(std::ofstream& file, const Mesh& mesh);

    /**
     * @brief Write point data (nodal fields)
     */
    void write_point_data(std::ofstream& file, const State& state);

    /**
     * @brief Write cell data (element fields)
     */
    void write_cell_data(std::ofstream& file, const State& state);

    /**
     * @brief Get VTK cell type ID
     */
    int get_vtk_cell_type(ElementType type) const;

    /**
     * @brief Encode binary data to base64
     */
    std::string encode_base64(const void* data, std::size_t size) const;

    std::string base_filename_;
    std::string output_dir_;
    bool ascii_mode_;

    // Time series tracking
    struct TimeStepInfo {
        Real time;
        std::string filename;
    };
    std::vector<TimeStepInfo> time_steps_;
};

// ============================================================================
// Simple ASCII VTK Writer (Legacy Format)
// ============================================================================

/**
 * @brief Simple VTK writer using legacy ASCII format
 *
 * Easier to implement and debug than XML format
 * Still compatible with ParaView
 */
class SimpleVTKWriter {
public:
    SimpleVTKWriter(const std::string& filename)
        : filename_(filename), output_deformed_(true) {}

    /**
     * @brief Write mesh and solution to legacy VTK format
     */
    void write(const Mesh& mesh, const State* state = nullptr);

    /**
     * @brief Write mesh only
     */
    void write_mesh(const Mesh& mesh);

    /**
     * @brief Add scalar field (per-node)
     */
    void add_scalar_field(const std::string& name, const std::vector<Real>& data);

    /**
     * @brief Add vector field (per-node)
     */
    void add_vector_field(const std::string& name, const std::vector<Real>& data);

    /**
     * @brief Add cell scalar field (per-element)
     */
    void add_cell_scalar_field(const std::string& name, const std::vector<Real>& data);

    /**
     * @brief Set whether to output deformed geometry
     */
    void set_output_deformed(bool deformed) { output_deformed_ = deformed; }

    /**
     * @brief Set stress data for output (6 components per element: xx, yy, zz, xy, yz, xz)
     */
    void set_stress_data(const std::vector<Real>& stress) { stress_data_ = stress; }

    /**
     * @brief Set strain data for output (6 components per element)
     */
    void set_strain_data(const std::vector<Real>& strain) { strain_data_ = strain; }

    /**
     * @brief Set effective plastic strain (per element)
     */
    void set_plastic_strain(const std::vector<Real>& eps_p) { plastic_strain_ = eps_p; }

private:
    /**
     * @brief Get VTK cell type ID
     */
    int get_vtk_cell_type(ElementType type) const;

    std::string filename_;
    bool output_deformed_;
    std::map<std::string, std::vector<Real>> scalar_fields_;
    std::map<std::string, std::vector<Real>> vector_fields_;
    std::map<std::string, std::vector<Real>> cell_scalar_fields_;

    // Element data
    std::vector<Real> stress_data_;
    std::vector<Real> strain_data_;
    std::vector<Real> plastic_strain_;
};

} // namespace io
} // namespace nxs
