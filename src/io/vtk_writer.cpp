/**
 * @file vtk_writer.cpp
 * @brief VTK writer implementation
 */

#include <nexussim/io/vtk_writer.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/physics/element.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nxs {
namespace io {

// ============================================================================
// Simple VTK Writer (Legacy ASCII Format)
// ============================================================================

void SimpleVTKWriter::write(const Mesh& mesh, const State* state) {
    std::ofstream file(filename_);
    if (!file.is_open()) {
        throw FileIOError("Failed to open file: " + filename_);
    }

    NXS_LOG_INFO("Writing VTK file: {}", filename_);

    // Header
    file << "# vtk DataFile Version 3.0\n";
    file << "NexusSim FEM Output\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Points (nodes) - output deformed geometry if displacement available
    const std::size_t num_nodes = mesh.num_nodes();
    file << "POINTS " << num_nodes << " double\n";

    const auto& coords = mesh.coordinates();

    // Check if we should output deformed geometry
    bool has_disp = (state != nullptr && state->has_field("displacement"));

    for (std::size_t i = 0; i < num_nodes; ++i) {
        Real x = coords.at(i, 0);
        Real y = coords.at(i, 1);
        Real z = coords.at(i, 2);

        // Add displacement to get deformed position
        if (has_disp && output_deformed_) {
            const auto& disp = state->field("displacement");
            x += disp.at(i, 0);
            y += disp.at(i, 1);
            z += disp.at(i, 2);
        }

        file << x << " " << y << " " << z << "\n";
    }

    // Cells (elements)
    const auto& element_blocks = mesh.element_blocks();

    // Count total elements and connectivity size
    std::size_t total_elems = 0;
    std::size_t connectivity_size = 0;
    for (const auto& block : element_blocks) {
        const std::size_t num_elems = block.num_elements();
        total_elems += num_elems;
        connectivity_size += num_elems * (block.num_nodes_per_elem + 1); // +1 for count
    }

    file << "\nCELLS " << total_elems << " " << connectivity_size << "\n";

    // Write connectivity
    for (const auto& block : element_blocks) {
        const std::size_t num_elems = block.num_elements();
        const std::size_t nodes_per_elem = block.num_nodes_per_elem;

        for (std::size_t e = 0; e < num_elems; ++e) {
            file << nodes_per_elem;
            auto elem_nodes = block.element_nodes(e);
            for (const auto& node : elem_nodes) {
                file << " " << node;
            }
            file << "\n";
        }
    }

    // Cell types
    file << "\nCELL_TYPES " << total_elems << "\n";
    for (const auto& block : element_blocks) {
        const int vtk_type = get_vtk_cell_type(block.type);
        for (std::size_t e = 0; e < block.num_elements(); ++e) {
            file << vtk_type << "\n";
        }
    }

    // Point data (nodal fields)
    if (state != nullptr) {
        file << "\nPOINT_DATA " << num_nodes << "\n";

        // Displacement
        if (state->has_field("displacement")) {
            const auto& disp = state->field("displacement");
            file << "VECTORS displacement double\n";
            for (std::size_t i = 0; i < num_nodes; ++i) {
                file << disp.at(i, 0) << " "
                     << disp.at(i, 1) << " "
                     << disp.at(i, 2) << "\n";
            }
        }

        // Velocity
        if (state->has_field("velocity")) {
            const auto& vel = state->field("velocity");
            file << "VECTORS velocity double\n";
            for (std::size_t i = 0; i < num_nodes; ++i) {
                file << vel.at(i, 0) << " "
                     << vel.at(i, 1) << " "
                     << vel.at(i, 2) << "\n";
            }
        }

        // Acceleration
        if (state->has_field("acceleration")) {
            const auto& acc = state->field("acceleration");
            file << "VECTORS acceleration double\n";
            for (std::size_t i = 0; i < num_nodes; ++i) {
                file << acc.at(i, 0) << " "
                     << acc.at(i, 1) << " "
                     << acc.at(i, 2) << "\n";
            }
        }

        // Velocity magnitude
        if (state->has_field("velocity")) {
            const auto& vel = state->field("velocity");
            file << "SCALARS velocity_magnitude double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (std::size_t i = 0; i < num_nodes; ++i) {
                const Real vx = vel.at(i, 0);
                const Real vy = vel.at(i, 1);
                const Real vz = vel.at(i, 2);
                const Real vmag = std::sqrt(vx*vx + vy*vy + vz*vz);
                file << vmag << "\n";
            }
        }

        // Displacement magnitude
        if (state->has_field("displacement")) {
            const auto& disp = state->field("displacement");
            file << "SCALARS displacement_magnitude double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (std::size_t i = 0; i < num_nodes; ++i) {
                const Real ux = disp.at(i, 0);
                const Real uy = disp.at(i, 1);
                const Real uz = disp.at(i, 2);
                const Real umag = std::sqrt(ux*ux + uy*uy + uz*uz);
                file << umag << "\n";
            }
        }

        // User-defined scalar fields
        for (const auto& [name, data] : scalar_fields_) {
            if (data.size() == num_nodes) {
                file << "SCALARS " << name << " double 1\n";
                file << "LOOKUP_TABLE default\n";
                for (const auto& val : data) {
                    file << val << "\n";
                }
            }
        }

        // User-defined vector fields
        for (const auto& [name, data] : vector_fields_) {
            if (data.size() == num_nodes * 3) {
                file << "VECTORS " << name << " double\n";
                for (std::size_t i = 0; i < num_nodes; ++i) {
                    file << data[i*3] << " " << data[i*3+1] << " " << data[i*3+2] << "\n";
                }
            }
        }
    }

    // Cell data (element-level fields)
    if (total_elems > 0 && (!stress_data_.empty() || !strain_data_.empty() ||
                            !plastic_strain_.empty() || !cell_scalar_fields_.empty())) {
        file << "\nCELL_DATA " << total_elems << "\n";

        // Von Mises stress from stress tensor
        if (stress_data_.size() == total_elems * 6) {
            file << "SCALARS von_mises_stress double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (std::size_t e = 0; e < total_elems; ++e) {
                const Real* s = &stress_data_[e * 6];
                // s = [sxx, syy, szz, sxy, syz, sxz]
                const Real sxx = s[0], syy = s[1], szz = s[2];
                const Real sxy = s[3], syz = s[4], sxz = s[5];

                // von Mises: sqrt(0.5*((sx-sy)^2 + (sy-sz)^2 + (sz-sx)^2 + 6*(sxy^2+syz^2+sxz^2)))
                const Real vm = std::sqrt(0.5 * ((sxx-syy)*(sxx-syy) + (syy-szz)*(syy-szz) +
                                                  (szz-sxx)*(szz-sxx) + 6.0*(sxy*sxy + syz*syz + sxz*sxz)));
                file << vm << "\n";
            }

            // Individual stress components
            const char* stress_names[] = {"stress_xx", "stress_yy", "stress_zz",
                                          "stress_xy", "stress_yz", "stress_xz"};
            for (int comp = 0; comp < 6; ++comp) {
                file << "SCALARS " << stress_names[comp] << " double 1\n";
                file << "LOOKUP_TABLE default\n";
                for (std::size_t e = 0; e < total_elems; ++e) {
                    file << stress_data_[e * 6 + comp] << "\n";
                }
            }
        }

        // Strain components
        if (strain_data_.size() == total_elems * 6) {
            const char* strain_names[] = {"strain_xx", "strain_yy", "strain_zz",
                                          "strain_xy", "strain_yz", "strain_xz"};
            for (int comp = 0; comp < 6; ++comp) {
                file << "SCALARS " << strain_names[comp] << " double 1\n";
                file << "LOOKUP_TABLE default\n";
                for (std::size_t e = 0; e < total_elems; ++e) {
                    file << strain_data_[e * 6 + comp] << "\n";
                }
            }
        }

        // Effective plastic strain
        if (plastic_strain_.size() == total_elems) {
            file << "SCALARS effective_plastic_strain double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (const auto& eps : plastic_strain_) {
                file << eps << "\n";
            }
        }

        // User-defined cell scalar fields
        for (const auto& [name, data] : cell_scalar_fields_) {
            if (data.size() == total_elems) {
                file << "SCALARS " << name << " double 1\n";
                file << "LOOKUP_TABLE default\n";
                for (const auto& val : data) {
                    file << val << "\n";
                }
            }
        }
    }

    file.close();
    NXS_LOG_INFO("VTK file written successfully");
}

void SimpleVTKWriter::write_mesh(const Mesh& mesh) {
    write(mesh, nullptr);
}

int SimpleVTKWriter::get_vtk_cell_type(ElementType type) const {
    // VTK cell type IDs
    // See: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    switch (type) {
        case ElementType::Hex8:
            return 12; // VTK_HEXAHEDRON
        case ElementType::Hex20:
            return 25; // VTK_QUADRATIC_HEXAHEDRON
        case ElementType::Tet4:
            return 10; // VTK_TETRA
        case ElementType::Tet10:
            return 24; // VTK_QUADRATIC_TETRA
        case ElementType::Wedge6:
            return 13; // VTK_WEDGE
        case ElementType::Shell4:
            return 9;  // VTK_QUAD
        case ElementType::Shell3:
            return 5;  // VTK_TRIANGLE
        case ElementType::Beam2:
            return 3;  // VTK_LINE
        default:
            NXS_LOG_WARN("Unknown element type for VTK, using VTK_VERTEX");
            return 1;  // VTK_VERTEX
    }
}

void SimpleVTKWriter::add_scalar_field(const std::string& name, const std::vector<Real>& data) {
    scalar_fields_[name] = data;
}

void SimpleVTKWriter::add_vector_field(const std::string& name, const std::vector<Real>& data) {
    vector_fields_[name] = data;
}

void SimpleVTKWriter::add_cell_scalar_field(const std::string& name, const std::vector<Real>& data) {
    cell_scalar_fields_[name] = data;
}

// ============================================================================
// XML VTK Writer (Full Implementation - Future)
// ============================================================================

VTKWriter::VTKWriter(const std::string& filename)
    : base_filename_(filename)
    , output_dir_(".")
    , ascii_mode_(false)
{
}

void VTKWriter::write(const Mesh& mesh, const State* state, Real time) {
    // For now, delegate to SimpleVTKWriter
    SimpleVTKWriter simple_writer(base_filename_ + ".vtk");
    simple_writer.write(mesh, state);
}

void VTKWriter::write_time_step(const Mesh& mesh, const State& state, Real time, int step) {
    std::ostringstream filename;
    filename << output_dir_ << "/" << base_filename_
             << "_" << std::setfill('0') << std::setw(6) << step << ".vtk";

    SimpleVTKWriter writer(filename.str());
    writer.write(mesh, &state);

    // Track for .pvd file
    TimeStepInfo info;
    info.time = time;
    info.filename = filename.str();
    time_steps_.push_back(info);

    NXS_LOG_INFO("Written time step {} at t={:.6e} s", step, time);
}

void VTKWriter::finalize_time_series() {
    if (time_steps_.empty()) {
        return;
    }

    std::string pvd_filename = output_dir_ + "/" + base_filename_ + ".pvd";
    std::ofstream file(pvd_filename);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <Collection>\n";

    for (const auto& ts : time_steps_) {
        file << "    <DataSet timestep=\"" << ts.time
             << "\" file=\"" << ts.filename << "\"/>\n";
    }

    file << "  </Collection>\n";
    file << "</VTKFile>\n";

    file.close();
    NXS_LOG_INFO("Written ParaView collection file: {}", pvd_filename);
    NXS_LOG_INFO("Open in ParaView to view time series animation");
}

void VTKWriter::write_header(std::ofstream& file) {
    // XML format header (future implementation)
}

void VTKWriter::write_points(std::ofstream& file, const Mesh& mesh) {
    // XML format points (future implementation)
}

void VTKWriter::write_cells(std::ofstream& file, const Mesh& mesh) {
    // XML format cells (future implementation)
}

void VTKWriter::write_point_data(std::ofstream& file, const State& state) {
    // XML format point data (future implementation)
}

void VTKWriter::write_cell_data(std::ofstream& file, const State& state) {
    // XML format cell data (future implementation)
}

std::string VTKWriter::encode_base64(const void* data, std::size_t size) const {
    // Base64 encoding (future implementation)
    return "";
}

} // namespace io
} // namespace nxs
