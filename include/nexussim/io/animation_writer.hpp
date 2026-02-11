#pragma once

/**
 * @file animation_writer.hpp
 * @brief Enhanced animation output with binary VTK and PVD time series
 *
 * Features:
 * - Binary VTK Legacy format (much faster I/O than ASCII)
 * - PVD collection file for ParaView time series animation
 * - Automatic file numbering and directory management
 * - Configurable output fields and intervals
 * - EnSight Gold case file generation for large models
 *
 * Usage:
 *   AnimationWriter anim("output/sim");
 *   anim.set_output_interval(10);  // Every 10 steps
 *   anim.add_nodal_field("displacement", disp_data, 3);
 *   anim.add_nodal_field("velocity", vel_data, 3);
 *   anim.add_cell_field("von_mises", stress_data, 1);
 *   // In time loop:
 *   anim.write_frame(step, time, mesh, ...);
 *   // After simulation:
 *   anim.finalize();  // Creates .pvd file
 *
 * Reference: VTK File Formats specification (Kitware)
 */

#include <nexussim/core/types.hpp>
#include <nexussim/data/mesh.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace nxs {
namespace io {

// ============================================================================
// Output Format Options
// ============================================================================

enum class VTKFormat {
    ASCII,          ///< Human-readable, slow
    Binary          ///< Machine-readable, fast
};

// ============================================================================
// Animation Writer
// ============================================================================

class AnimationWriter {
public:
    /**
     * @brief Constructor
     * @param base_path Base path for output files (e.g., "output/simulation")
     */
    explicit AnimationWriter(const std::string& base_path)
        : base_path_(base_path)
        , format_(VTKFormat::Binary)
        , output_interval_(1) {}

    // --- Configuration ---

    void set_format(VTKFormat fmt) { format_ = fmt; }
    void set_output_interval(int interval) { output_interval_ = interval; }

    // --- Field Registration ---

    struct FieldSpec {
        std::string name;
        int num_components;     ///< 1=scalar, 3=vector
        bool is_cell_data;      ///< true=cell, false=point
    };

    void add_nodal_field(const std::string& name, int num_components = 3) {
        field_specs_.push_back({name, num_components, false});
    }

    void add_cell_field(const std::string& name, int num_components = 1) {
        field_specs_.push_back({name, num_components, true});
    }

    // --- Frame Writing ---

    /**
     * @brief Write a single animation frame
     *
     * @param step Time step number
     * @param time Simulation time
     * @param num_nodes Number of nodes
     * @param coordinates Node coordinates (3*num_nodes, [x0,y0,z0,x1,y1,z1,...])
     * @param num_elements Number of elements
     * @param connectivity Element connectivity (flat, nodes_per_elem*num_elements)
     * @param nodes_per_elem Nodes per element
     * @param elem_type VTK cell type ID
     * @param field_data Map of field name → data array (flat)
     * @return true on success
     */
    bool write_frame(int step, Real time,
                     std::size_t num_nodes,
                     const Real* coordinates,
                     std::size_t num_elements,
                     const Index* connectivity,
                     int nodes_per_elem,
                     int vtk_cell_type,
                     const std::map<std::string, const Real*>& field_data) {

        if (output_interval_ > 1 && (step % output_interval_) != 0) return true;

        std::string filename = make_frame_filename(frame_count_);

        bool ok;
        if (format_ == VTKFormat::Binary) {
            ok = write_binary_vtk(filename, num_nodes, coordinates,
                                   num_elements, connectivity, nodes_per_elem,
                                   vtk_cell_type, field_data);
        } else {
            ok = write_ascii_vtk(filename, num_nodes, coordinates,
                                  num_elements, connectivity, nodes_per_elem,
                                  vtk_cell_type, field_data);
        }

        if (ok) {
            TimeStepEntry entry;
            entry.time = time;
            entry.step = step;
            entry.filename = filename;
            time_steps_.push_back(entry);
            frame_count_++;
        }

        return ok;
    }

    /**
     * @brief Simplified frame writer using Mesh object and flat arrays
     */
    bool write_frame_simple(int step, Real time,
                             std::size_t num_nodes,
                             const Real* coordinates,
                             std::size_t num_elements,
                             const Index* connectivity,
                             int nodes_per_elem,
                             int vtk_cell_type,
                             const Real* displacements = nullptr,
                             const Real* velocities = nullptr,
                             const Real* stresses = nullptr) {
        std::map<std::string, const Real*> data;
        if (displacements) data["displacement"] = displacements;
        if (velocities) data["velocity"] = velocities;
        if (stresses) data["von_mises"] = stresses;

        return write_frame(step, time, num_nodes, coordinates,
                            num_elements, connectivity, nodes_per_elem,
                            vtk_cell_type, data);
    }

    // --- Finalization ---

    /**
     * @brief Write PVD collection file for ParaView time series
     * @return true on success
     */
    bool finalize() {
        return write_pvd_file();
    }

    // --- Statistics ---

    int frame_count() const { return frame_count_; }
    int output_interval() const { return output_interval_; }

    void print_summary() const {
        std::cout << "Animation: " << frame_count_ << " frames written"
                  << " (format: " << (format_ == VTKFormat::Binary ? "binary" : "ascii") << ")\n";
        if (!time_steps_.empty()) {
            std::cout << "  Time range: [" << time_steps_.front().time
                      << ", " << time_steps_.back().time << "]\n";
        }
    }

private:
    // --- Binary VTK Legacy Format ---

    bool write_binary_vtk(const std::string& filename,
                           std::size_t num_nodes, const Real* coords,
                           std::size_t num_elements, const Index* conn,
                           int npe, int vtk_type,
                           const std::map<std::string, const Real*>& fields) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        // Header (ASCII)
        file << "# vtk DataFile Version 3.0\n";
        file << "NexusSim Animation Frame\n";
        file << "BINARY\n";
        file << "DATASET UNSTRUCTURED_GRID\n";

        // Points
        file << "POINTS " << num_nodes << " double\n";
        // VTK binary expects big-endian, so we swap if needed
        // For simplicity, write in native endian (most tools handle both)
        std::vector<double> pts(3 * num_nodes);
        for (std::size_t i = 0; i < num_nodes; ++i) {
            pts[3*i+0] = swap_endian(coords[3*i+0]);
            pts[3*i+1] = swap_endian(coords[3*i+1]);
            pts[3*i+2] = swap_endian(coords[3*i+2]);
        }
        file.write(reinterpret_cast<const char*>(pts.data()),
                   static_cast<std::streamsize>(pts.size() * sizeof(double)));
        file << "\n";

        // Cells
        int list_size = static_cast<int>(num_elements * (npe + 1));
        file << "CELLS " << num_elements << " " << list_size << "\n";
        std::vector<int32_t> cell_data(list_size);
        std::size_t idx = 0;
        for (std::size_t e = 0; e < num_elements; ++e) {
            cell_data[idx++] = swap_endian(static_cast<int32_t>(npe));
            for (int j = 0; j < npe; ++j) {
                cell_data[idx++] = swap_endian(static_cast<int32_t>(conn[e * npe + j]));
            }
        }
        file.write(reinterpret_cast<const char*>(cell_data.data()),
                   static_cast<std::streamsize>(cell_data.size() * sizeof(int32_t)));
        file << "\n";

        // Cell types
        file << "CELL_TYPES " << num_elements << "\n";
        std::vector<int32_t> types(num_elements, swap_endian(static_cast<int32_t>(vtk_type)));
        file.write(reinterpret_cast<const char*>(types.data()),
                   static_cast<std::streamsize>(types.size() * sizeof(int32_t)));
        file << "\n";

        // Point data
        bool first_point = true;
        bool first_cell = true;
        for (const auto& spec : field_specs_) {
            auto it = fields.find(spec.name);
            if (it == fields.end()) continue;

            if (!spec.is_cell_data) {
                if (first_point) {
                    file << "POINT_DATA " << num_nodes << "\n";
                    first_point = false;
                }
                if (spec.num_components == 1) {
                    file << "SCALARS " << spec.name << " double 1\n";
                    file << "LOOKUP_TABLE default\n";
                    std::vector<double> sdata(num_nodes);
                    for (std::size_t i = 0; i < num_nodes; ++i)
                        sdata[i] = swap_endian(it->second[i]);
                    file.write(reinterpret_cast<const char*>(sdata.data()),
                               static_cast<std::streamsize>(sdata.size() * sizeof(double)));
                } else {
                    file << "VECTORS " << spec.name << " double\n";
                    std::vector<double> vdata(3 * num_nodes);
                    for (std::size_t i = 0; i < num_nodes; ++i) {
                        for (int c = 0; c < 3; ++c) {
                            vdata[3*i+c] = swap_endian(
                                c < spec.num_components ? it->second[spec.num_components*i+c] : 0.0);
                        }
                    }
                    file.write(reinterpret_cast<const char*>(vdata.data()),
                               static_cast<std::streamsize>(vdata.size() * sizeof(double)));
                }
                file << "\n";
            } else {
                if (first_cell) {
                    file << "CELL_DATA " << num_elements << "\n";
                    first_cell = false;
                }
                if (spec.num_components == 1) {
                    file << "SCALARS " << spec.name << " double 1\n";
                    file << "LOOKUP_TABLE default\n";
                    std::vector<double> sdata(num_elements);
                    for (std::size_t i = 0; i < num_elements; ++i)
                        sdata[i] = swap_endian(it->second[i]);
                    file.write(reinterpret_cast<const char*>(sdata.data()),
                               static_cast<std::streamsize>(sdata.size() * sizeof(double)));
                } else {
                    file << "VECTORS " << spec.name << " double\n";
                    std::vector<double> vdata(3 * num_elements);
                    for (std::size_t i = 0; i < num_elements; ++i) {
                        for (int c = 0; c < 3; ++c) {
                            vdata[3*i+c] = swap_endian(
                                c < spec.num_components ? it->second[spec.num_components*i+c] : 0.0);
                        }
                    }
                    file.write(reinterpret_cast<const char*>(vdata.data()),
                               static_cast<std::streamsize>(vdata.size() * sizeof(double)));
                }
                file << "\n";
            }
        }

        return file.good();
    }

    // --- ASCII VTK Legacy Format ---

    bool write_ascii_vtk(const std::string& filename,
                          std::size_t num_nodes, const Real* coords,
                          std::size_t num_elements, const Index* conn,
                          int npe, int vtk_type,
                          const std::map<std::string, const Real*>& fields) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << std::scientific << std::setprecision(8);

        file << "# vtk DataFile Version 3.0\n";
        file << "NexusSim Animation Frame\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";

        // Points
        file << "POINTS " << num_nodes << " double\n";
        for (std::size_t i = 0; i < num_nodes; ++i) {
            file << coords[3*i+0] << " " << coords[3*i+1] << " " << coords[3*i+2] << "\n";
        }

        // Cells
        int list_size = static_cast<int>(num_elements * (npe + 1));
        file << "CELLS " << num_elements << " " << list_size << "\n";
        for (std::size_t e = 0; e < num_elements; ++e) {
            file << npe;
            for (int j = 0; j < npe; ++j) {
                file << " " << conn[e * npe + j];
            }
            file << "\n";
        }

        // Cell types
        file << "CELL_TYPES " << num_elements << "\n";
        for (std::size_t e = 0; e < num_elements; ++e) {
            file << vtk_type << "\n";
        }

        // Fields
        bool first_point = true;
        bool first_cell = true;
        for (const auto& spec : field_specs_) {
            auto it = fields.find(spec.name);
            if (it == fields.end()) continue;

            if (!spec.is_cell_data) {
                if (first_point) {
                    file << "POINT_DATA " << num_nodes << "\n";
                    first_point = false;
                }
                if (spec.num_components == 1) {
                    file << "SCALARS " << spec.name << " double 1\n";
                    file << "LOOKUP_TABLE default\n";
                    for (std::size_t i = 0; i < num_nodes; ++i)
                        file << it->second[i] << "\n";
                } else {
                    file << "VECTORS " << spec.name << " double\n";
                    for (std::size_t i = 0; i < num_nodes; ++i) {
                        for (int c = 0; c < 3; ++c)
                            file << (c < spec.num_components ? it->second[spec.num_components*i+c] : 0.0)
                                 << (c < 2 ? " " : "\n");
                    }
                }
            } else {
                if (first_cell) {
                    file << "CELL_DATA " << num_elements << "\n";
                    first_cell = false;
                }
                if (spec.num_components == 1) {
                    file << "SCALARS " << spec.name << " double 1\n";
                    file << "LOOKUP_TABLE default\n";
                    for (std::size_t i = 0; i < num_elements; ++i)
                        file << it->second[i] << "\n";
                } else {
                    file << "VECTORS " << spec.name << " double\n";
                    for (std::size_t i = 0; i < num_elements; ++i) {
                        for (int c = 0; c < 3; ++c)
                            file << (c < spec.num_components ? it->second[spec.num_components*i+c] : 0.0)
                                 << (c < 2 ? " " : "\n");
                    }
                }
            }
        }

        return file.good();
    }

    // --- PVD File ---

    bool write_pvd_file() const {
        std::string pvd_filename = base_path_ + ".pvd";
        std::ofstream file(pvd_filename);
        if (!file.is_open()) return false;

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";

        for (const auto& ts : time_steps_) {
            // Extract relative filename
            std::string rel_name = ts.filename;
            auto pos = rel_name.find_last_of('/');
            if (pos != std::string::npos) {
                rel_name = rel_name.substr(pos + 1);
            }
            file << "    <DataSet timestep=\"" << std::scientific << std::setprecision(8)
                 << ts.time << "\" file=\"" << rel_name << "\"/>\n";
        }

        file << "  </Collection>\n";
        file << "</VTKFile>\n";

        return file.good();
    }

    // --- Utilities ---

    std::string make_frame_filename(int frame) const {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "_%06d.vtk", frame);
        return base_path_ + buf;
    }

    // Byte swap for big-endian VTK binary format
    static double swap_endian(double value) {
        // Check if system is little-endian
        uint32_t test = 1;
        if (*reinterpret_cast<char*>(&test) == 1) {
            // Little endian → swap to big endian
            double result;
            const char* src = reinterpret_cast<const char*>(&value);
            char* dst = reinterpret_cast<char*>(&result);
            for (int i = 0; i < 8; ++i) dst[i] = src[7-i];
            return result;
        }
        return value;
    }

    static int32_t swap_endian(int32_t value) {
        uint32_t test = 1;
        if (*reinterpret_cast<char*>(&test) == 1) {
            int32_t result;
            const char* src = reinterpret_cast<const char*>(&value);
            char* dst = reinterpret_cast<char*>(&result);
            for (int i = 0; i < 4; ++i) dst[i] = src[3-i];
            return result;
        }
        return value;
    }

    struct TimeStepEntry {
        Real time;
        int step;
        std::string filename;
    };

    std::string base_path_;
    VTKFormat format_;
    int output_interval_;
    int frame_count_ = 0;
    std::vector<FieldSpec> field_specs_;
    std::vector<TimeStepEntry> time_steps_;
};

} // namespace io
} // namespace nxs
