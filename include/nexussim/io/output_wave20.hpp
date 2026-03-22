#pragma once

/**
 * @file output_wave20.hpp
 * @brief Wave 20: Advanced Output Format Writers
 *
 * Six output format writers for post-processing interoperability:
 *  1. BinaryAnimationWriter  - Compact seekable binary animation format
 *  2. H3DWriter              - Altair HyperView compatible output
 *  3. D3PLOTWriter           - LS-PrePost compatible binary output
 *  4. EnSightGoldWriter      - EnSight Gold case/geometry/variable format
 *  5. TimeHistoryExporter    - CSV and binary time series at probe locations
 *  6. CrossSectionForceOutput - Cutting plane force/moment resultants
 *
 * Reference: LS-DYNA D3PLOT format, EnSight Gold User Manual,
 *            Altair H3D specification, VTK binary format.
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

namespace nxs {
namespace io {

// ============================================================================
// Common data structures used across writers
// ============================================================================

/// Nodal data for a single frame (positions, displacements, velocities)
struct NodalFrameData {
    std::vector<Real> x, y, z;           ///< Current positions
    std::vector<Real> dx, dy, dz;        ///< Displacements
    std::vector<Real> vx, vy, vz;        ///< Velocities

    int num_nodes() const { return static_cast<int>(x.size()); }
};

/// Element data for a single frame (stress, strain, damage)
struct ElementFrameData {
    std::vector<Real> stress_xx, stress_yy, stress_zz;
    std::vector<Real> stress_xy, stress_yz, stress_xz;
    std::vector<Real> strain_xx, strain_yy, strain_zz;
    std::vector<Real> strain_xy, strain_yz, strain_xz;
    std::vector<Real> von_mises;
    std::vector<Real> plastic_strain;
    std::vector<Real> damage;

    int num_elements() const { return static_cast<int>(stress_xx.size()); }
};

// ############################################################################
// 1. BinaryAnimationWriter -- Compact seekable binary animation format
// ############################################################################

/**
 * Binary layout:
 *   [FileHeader]
 *   [FrameIndex: num_frames x FrameIndexEntry]  (seek table, written at close)
 *   [Frame 0: FrameHeader + nodal data + element data]
 *   [Frame 1: ...]
 *   ...
 *
 * File header occupies a fixed 64-byte block so the frame index position
 * is always known. The frame index is written at the end and its offset
 * stored in the header on close().
 */
class BinaryAnimationWriter {
public:
    static constexpr uint32_t MAGIC = 0x4E585342;  // "NXSB"
    static constexpr uint16_t VERSION_MAJOR = 1;
    static constexpr uint16_t VERSION_MINOR = 0;
    static constexpr int HEADER_SIZE = 64;

    struct FileHeader {
        uint32_t magic;
        uint16_t version_major;
        uint16_t version_minor;
        int32_t  num_frames;
        int32_t  num_nodes;
        int32_t  num_elements;
        int64_t  frame_index_offset;   ///< Byte offset to the frame index table
        int32_t  nodal_fields;         ///< Bitmask: 1=pos, 2=disp, 4=vel
        int32_t  elem_fields;          ///< Bitmask: 1=stress, 2=strain, 4=damage
        char     reserved[24];

        FileHeader()
            : magic(MAGIC), version_major(VERSION_MAJOR), version_minor(VERSION_MINOR)
            , num_frames(0), num_nodes(0), num_elements(0)
            , frame_index_offset(0), nodal_fields(7), elem_fields(7) {
            std::memset(reserved, 0, sizeof(reserved));
        }
    };

    struct FrameHeader {
        Real     time;
        int64_t  byte_offset;          ///< Absolute offset in file
        int32_t  nodal_bytes;          ///< Size of nodal data block
        int32_t  elem_bytes;           ///< Size of element data block
    };

    BinaryAnimationWriter() : is_open_(false) {}

    ~BinaryAnimationWriter() {
        if (is_open_) close();
    }

    /// Open the binary file for writing
    bool open(const std::string& filename, int num_nodes, int num_elements) {
        file_.open(filename, std::ios::binary | std::ios::trunc);
        if (!file_.is_open()) return false;
        is_open_ = true;
        filename_ = filename;

        header_.num_nodes = num_nodes;
        header_.num_elements = num_elements;
        header_.num_frames = 0;

        // Write placeholder header (will be updated on close)
        write_header();
        return true;
    }

    /// Write file header at position 0
    void write_header() {
        file_.seekp(0, std::ios::beg);
        file_.write(reinterpret_cast<const char*>(&header_.magic), sizeof(header_.magic));
        file_.write(reinterpret_cast<const char*>(&header_.version_major), sizeof(header_.version_major));
        file_.write(reinterpret_cast<const char*>(&header_.version_minor), sizeof(header_.version_minor));
        file_.write(reinterpret_cast<const char*>(&header_.num_frames), sizeof(header_.num_frames));
        file_.write(reinterpret_cast<const char*>(&header_.num_nodes), sizeof(header_.num_nodes));
        file_.write(reinterpret_cast<const char*>(&header_.num_elements), sizeof(header_.num_elements));
        file_.write(reinterpret_cast<const char*>(&header_.frame_index_offset), sizeof(header_.frame_index_offset));
        file_.write(reinterpret_cast<const char*>(&header_.nodal_fields), sizeof(header_.nodal_fields));
        file_.write(reinterpret_cast<const char*>(&header_.elem_fields), sizeof(header_.elem_fields));
        file_.write(header_.reserved, sizeof(header_.reserved));
    }

    /// Write a single animation frame
    bool write_frame(Real time, const NodalFrameData& node_data,
                     const ElementFrameData& elem_data) {
        if (!is_open_) return false;

        FrameHeader fh;
        fh.time = time;
        fh.byte_offset = static_cast<int64_t>(file_.tellp());

        int nn = header_.num_nodes;
        int ne = header_.num_elements;

        // Nodal data: positions(3*nn) + displacements(3*nn) + velocities(3*nn) = 9*nn doubles
        fh.nodal_bytes = static_cast<int32_t>(9 * nn * sizeof(Real));
        // Element data: stress(6*ne) + strain(6*ne) + von_mises(ne) + plastic_strain(ne) + damage(ne) = 15*ne
        fh.elem_bytes = static_cast<int32_t>(15 * ne * sizeof(Real));

        // Write frame header
        file_.write(reinterpret_cast<const char*>(&fh.time), sizeof(fh.time));
        file_.write(reinterpret_cast<const char*>(&fh.nodal_bytes), sizeof(fh.nodal_bytes));
        file_.write(reinterpret_cast<const char*>(&fh.elem_bytes), sizeof(fh.elem_bytes));

        // Write nodal data (interleaved by component for better compression)
        write_vector(node_data.x, nn);
        write_vector(node_data.y, nn);
        write_vector(node_data.z, nn);
        write_vector(node_data.dx, nn);
        write_vector(node_data.dy, nn);
        write_vector(node_data.dz, nn);
        write_vector(node_data.vx, nn);
        write_vector(node_data.vy, nn);
        write_vector(node_data.vz, nn);

        // Write element data
        write_vector(elem_data.stress_xx, ne);
        write_vector(elem_data.stress_yy, ne);
        write_vector(elem_data.stress_zz, ne);
        write_vector(elem_data.stress_xy, ne);
        write_vector(elem_data.stress_yz, ne);
        write_vector(elem_data.stress_xz, ne);
        write_vector(elem_data.strain_xx, ne);
        write_vector(elem_data.strain_yy, ne);
        write_vector(elem_data.strain_zz, ne);
        write_vector(elem_data.strain_xy, ne);
        write_vector(elem_data.strain_yz, ne);
        write_vector(elem_data.strain_xz, ne);
        write_vector(elem_data.von_mises, ne);
        write_vector(elem_data.plastic_strain, ne);
        write_vector(elem_data.damage, ne);

        frame_index_.push_back(fh);
        header_.num_frames++;
        return true;
    }

    /// Finalize: write frame index and update header
    void close() {
        if (!is_open_) return;

        // Record frame index offset
        header_.frame_index_offset = static_cast<int64_t>(file_.tellp());

        // Write frame index table
        int32_t nf = static_cast<int32_t>(frame_index_.size());
        file_.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
        for (const auto& fh : frame_index_) {
            file_.write(reinterpret_cast<const char*>(&fh.time), sizeof(fh.time));
            file_.write(reinterpret_cast<const char*>(&fh.byte_offset), sizeof(fh.byte_offset));
            file_.write(reinterpret_cast<const char*>(&fh.nodal_bytes), sizeof(fh.nodal_bytes));
            file_.write(reinterpret_cast<const char*>(&fh.elem_bytes), sizeof(fh.elem_bytes));
        }

        // Rewrite header with final frame count and index offset
        write_header();

        file_.close();
        is_open_ = false;
    }

    int num_frames() const { return header_.num_frames; }
    bool is_open() const { return is_open_; }

private:
    std::ofstream file_;
    std::string filename_;
    FileHeader header_;
    std::vector<FrameHeader> frame_index_;
    bool is_open_;

    void write_vector(const std::vector<Real>& v, int expected_size) {
        if (static_cast<int>(v.size()) >= expected_size) {
            file_.write(reinterpret_cast<const char*>(v.data()),
                        expected_size * sizeof(Real));
        } else {
            // Pad with zeros if data is shorter than expected
            file_.write(reinterpret_cast<const char*>(v.data()),
                        v.size() * sizeof(Real));
            std::vector<Real> pad(expected_size - static_cast<int>(v.size()), 0.0);
            file_.write(reinterpret_cast<const char*>(pad.data()),
                        pad.size() * sizeof(Real));
        }
    }
};

// ############################################################################
// 2. H3DWriter -- Altair HyperView compatible binary output
// ############################################################################

/**
 * H3D binary layout (simplified):
 *   [H3DFileHeader]
 *   [DataTypeBlock: NODE]
 *   [DataTypeBlock: ELEMENT]
 *   For each subcase (time step):
 *     [SubcaseHeader]
 *     [ResultRecord: node displacements]
 *     [ResultRecord: node velocities]
 *     [ResultRecord: element stress]
 *     [ResultRecord: element strain]
 */
class H3DWriter {
public:
    static constexpr uint32_t H3D_MAGIC = 0x48334400;  // "H3D\0"
    static constexpr int32_t H3D_VERSION = 11;
    static constexpr int TITLE_LEN = 80;

    enum class DataType : int32_t {
        NODE = 1,
        ELEMENT = 2
    };

    enum class ResultType : int32_t {
        DISPLACEMENT = 1,
        VELOCITY = 2,
        STRESS = 3,
        STRAIN = 4
    };

    struct H3DFileHeader {
        uint32_t magic;
        int32_t  version;
        char     title[TITLE_LEN];
        int32_t  num_nodes;
        int32_t  num_elements;
        int32_t  num_subcases;

        H3DFileHeader()
            : magic(H3D_MAGIC), version(H3D_VERSION)
            , num_nodes(0), num_elements(0), num_subcases(0) {
            std::memset(title, 0, TITLE_LEN);
        }
    };

    struct SubcaseHeader {
        int32_t  subcase_id;
        Real     time;
        int32_t  num_results;
    };

    struct ResultRecord {
        ResultType type;
        DataType   data_type;
        int32_t    num_components;
        int32_t    num_entries;
        // Followed by num_entries * num_components * sizeof(Real) bytes
    };

    H3DWriter() : is_open_(false), subcase_open_(false), current_subcase_results_(0) {}

    ~H3DWriter() {
        if (subcase_open_) end_subcase();
        if (is_open_) close();
    }

    /// Open H3D file for writing
    bool open(const std::string& filename, const std::string& title,
              int num_nodes, int num_elements) {
        file_.open(filename, std::ios::binary | std::ios::trunc);
        if (!file_.is_open()) return false;
        is_open_ = true;

        header_.num_nodes = num_nodes;
        header_.num_elements = num_elements;
        header_.num_subcases = 0;

        // Copy title (truncate if needed)
        std::memset(header_.title, 0, TITLE_LEN);
        int len = std::min(static_cast<int>(title.size()), TITLE_LEN - 1);
        std::memcpy(header_.title, title.c_str(), len);

        // Write placeholder header
        write_file_header();

        // Write datatype blocks
        write_datatype_block(DataType::NODE, num_nodes);
        write_datatype_block(DataType::ELEMENT, num_elements);

        return true;
    }

    /// Begin a new subcase (time step)
    bool begin_subcase(Real time) {
        if (!is_open_ || subcase_open_) return false;
        subcase_open_ = true;
        current_subcase_results_ = 0;

        SubcaseHeader sh;
        sh.subcase_id = header_.num_subcases + 1;
        sh.time = time;
        sh.num_results = 0;  // Will be updated by end_subcase

        subcase_header_offset_ = file_.tellp();
        file_.write(reinterpret_cast<const char*>(&sh.subcase_id), sizeof(sh.subcase_id));
        file_.write(reinterpret_cast<const char*>(&sh.time), sizeof(sh.time));
        file_.write(reinterpret_cast<const char*>(&sh.num_results), sizeof(sh.num_results));

        return true;
    }

    /// Write node displacement and velocity results
    void write_node_results(const std::vector<Real>& displacements,
                            const std::vector<Real>& velocities) {
        if (!subcase_open_) return;

        int nn = header_.num_nodes;

        // Displacements (3 components per node)
        if (static_cast<int>(displacements.size()) >= nn * 3) {
            write_result_record(ResultType::DISPLACEMENT, DataType::NODE, 3, nn,
                                displacements.data());
            current_subcase_results_++;
        }

        // Velocities (3 components per node)
        if (static_cast<int>(velocities.size()) >= nn * 3) {
            write_result_record(ResultType::VELOCITY, DataType::NODE, 3, nn,
                                velocities.data());
            current_subcase_results_++;
        }
    }

    /// Write element stress and strain results
    void write_element_results(const std::vector<Real>& stress,
                               const std::vector<Real>& strain) {
        if (!subcase_open_) return;

        int ne = header_.num_elements;

        // Stress (6 Voigt components per element)
        if (static_cast<int>(stress.size()) >= ne * 6) {
            write_result_record(ResultType::STRESS, DataType::ELEMENT, 6, ne,
                                stress.data());
            current_subcase_results_++;
        }

        // Strain (6 Voigt components per element)
        if (static_cast<int>(strain.size()) >= ne * 6) {
            write_result_record(ResultType::STRAIN, DataType::ELEMENT, 6, ne,
                                strain.data());
            current_subcase_results_++;
        }
    }

    /// End the current subcase, update its result count
    void end_subcase() {
        if (!subcase_open_) return;
        subcase_open_ = false;
        header_.num_subcases++;

        // Seek back to subcase header to update num_results
        auto current_pos = file_.tellp();
        file_.seekp(subcase_header_offset_ +
                     static_cast<std::streamoff>(sizeof(int32_t) + sizeof(Real)),
                     std::ios::beg);
        file_.write(reinterpret_cast<const char*>(&current_subcase_results_),
                    sizeof(current_subcase_results_));
        file_.seekp(current_pos, std::ios::beg);
    }

    /// Close file, update header with final subcase count
    void close() {
        if (!is_open_) return;
        if (subcase_open_) end_subcase();

        // Rewrite header with final counts
        write_file_header();

        file_.close();
        is_open_ = false;
    }

    int num_subcases() const { return header_.num_subcases; }
    bool is_open() const { return is_open_; }

private:
    std::ofstream file_;
    H3DFileHeader header_;
    bool is_open_;
    bool subcase_open_;
    int32_t current_subcase_results_;
    std::streampos subcase_header_offset_;

    void write_file_header() {
        file_.seekp(0, std::ios::beg);
        file_.write(reinterpret_cast<const char*>(&header_.magic), sizeof(header_.magic));
        file_.write(reinterpret_cast<const char*>(&header_.version), sizeof(header_.version));
        file_.write(header_.title, TITLE_LEN);
        file_.write(reinterpret_cast<const char*>(&header_.num_nodes), sizeof(header_.num_nodes));
        file_.write(reinterpret_cast<const char*>(&header_.num_elements), sizeof(header_.num_elements));
        file_.write(reinterpret_cast<const char*>(&header_.num_subcases), sizeof(header_.num_subcases));
    }

    void write_datatype_block(DataType dt, int32_t count) {
        int32_t dt_val = static_cast<int32_t>(dt);
        file_.write(reinterpret_cast<const char*>(&dt_val), sizeof(dt_val));
        file_.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }

    void write_result_record(ResultType rt, DataType dt, int32_t ncomp,
                             int32_t nentries, const Real* data) {
        ResultRecord rec;
        rec.type = rt;
        rec.data_type = dt;
        rec.num_components = ncomp;
        rec.num_entries = nentries;
        file_.write(reinterpret_cast<const char*>(&rec.type), sizeof(rec.type));
        file_.write(reinterpret_cast<const char*>(&rec.data_type), sizeof(rec.data_type));
        file_.write(reinterpret_cast<const char*>(&rec.num_components), sizeof(rec.num_components));
        file_.write(reinterpret_cast<const char*>(&rec.num_entries), sizeof(rec.num_entries));
        file_.write(reinterpret_cast<const char*>(data),
                    nentries * ncomp * sizeof(Real));
    }
};

// ############################################################################
// 3. D3PLOTWriter -- LS-PrePost compatible binary output
// ############################################################################

/**
 * Simplified D3PLOT binary layout:
 *   [Control words: 64 int32 header]
 *   [Geometry section: node coords + element connectivity]
 *   [State 0: time + nodal disp + elem stress]
 *   [State 1: ...]
 *   ...
 *
 * Control words encode metadata: num_nodes, num_elements, ndim, nv1d/2d/3d, etc.
 * All multi-byte values are written in native byte order (typically little-endian).
 * Word size is 4 bytes for int32 and 8 bytes for double (controlled by ICODE).
 */
class D3PLOTWriter {
public:
    static constexpr int CONTROL_WORDS = 64;
    static constexpr int WORD_SIZE = 4;        ///< 4-byte word (int32/float)

    D3PLOTWriter() : is_open_(false), num_nodes_(0), num_elements_(0),
                     num_states_(0), use_double_(true) {}

    ~D3PLOTWriter() {
        if (is_open_) close();
    }

    /// Open the D3PLOT file
    bool open(const std::string& filename, bool double_precision = true) {
        file_.open(filename, std::ios::binary | std::ios::trunc);
        if (!file_.is_open()) return false;
        is_open_ = true;
        filename_ = filename;
        use_double_ = double_precision;
        num_states_ = 0;
        return true;
    }

    /// Write geometry section: node coordinates and element connectivity
    /// nodes: flat array [x0,y0,z0, x1,y1,z1, ...]
    /// elements: flat array of node indices per element (hex8 = 8 per elem)
    void write_geometry(const std::vector<Real>& nodes,
                        const std::vector<int32_t>& elements,
                        int nodes_per_element = 8) {
        if (!is_open_) return;

        num_nodes_ = static_cast<int32_t>(nodes.size() / 3);
        int32_t nelem = static_cast<int32_t>(elements.size() / nodes_per_element);
        num_elements_ = nelem;

        // Write control words (64 x int32)
        int32_t control[CONTROL_WORDS];
        std::memset(control, 0, sizeof(control));
        control[0]  = 0;                       // NDIM placeholder
        control[1]  = num_nodes_;              // NUMNP
        control[2]  = 0;                       // IT (time step)
        control[3]  = nelem;                   // NEL8 (8-node solid elements)
        control[4]  = 0;                       // NUMMAT
        control[5]  = 0;                       // NV1D
        control[6]  = 0;                       // NV2D
        control[7]  = 7;                       // NV3D (stress components: 7 per element)
        control[8]  = 3;                       // NDIM (3D)
        control[9]  = nodes_per_element;       // Nodes per element
        control[10] = use_double_ ? 8 : 4;    // Word size for reals
        control[11] = 0;                       // Number of 1D elements
        control[12] = 0;                       // Number of 2D elements

        file_.write(reinterpret_cast<const char*>(control), sizeof(control));

        // Write node coordinates
        if (use_double_) {
            file_.write(reinterpret_cast<const char*>(nodes.data()),
                        num_nodes_ * 3 * sizeof(Real));
        } else {
            // Convert to float
            std::vector<float> fnodes(num_nodes_ * 3);
            for (int i = 0; i < num_nodes_ * 3; ++i)
                fnodes[i] = static_cast<float>(nodes[i]);
            file_.write(reinterpret_cast<const char*>(fnodes.data()),
                        num_nodes_ * 3 * sizeof(float));
        }

        // Write element connectivity (1-based indices in D3PLOT)
        std::vector<int32_t> conn_1based(elements.size());
        for (size_t i = 0; i < elements.size(); ++i)
            conn_1based[i] = elements[i] + 1;  // Convert 0-based to 1-based
        file_.write(reinterpret_cast<const char*>(conn_1based.data()),
                    conn_1based.size() * sizeof(int32_t));
    }

    /// Write a state section (one time step)
    /// nodal_disp: flat [dx0,dy0,dz0, dx1,dy1,dz1, ...] (3 per node)
    /// elem_stress: flat [sxx,syy,szz,sxy,syz,sxz,vm, ...] (7 per element)
    void write_state(Real time, const std::vector<Real>& nodal_disp,
                     const std::vector<Real>& elem_stress) {
        if (!is_open_) return;

        // State header: time value
        if (use_double_) {
            file_.write(reinterpret_cast<const char*>(&time), sizeof(Real));
        } else {
            float ftime = static_cast<float>(time);
            file_.write(reinterpret_cast<const char*>(&ftime), sizeof(float));
        }

        // Nodal displacements (3 components per node)
        int nd = num_nodes_ * 3;
        if (use_double_) {
            if (static_cast<int>(nodal_disp.size()) >= nd) {
                file_.write(reinterpret_cast<const char*>(nodal_disp.data()),
                            nd * sizeof(Real));
            } else {
                write_padded_double(nodal_disp, nd);
            }
        } else {
            std::vector<float> fd(nd, 0.0f);
            for (int i = 0; i < std::min(static_cast<int>(nodal_disp.size()), nd); ++i)
                fd[i] = static_cast<float>(nodal_disp[i]);
            file_.write(reinterpret_cast<const char*>(fd.data()), nd * sizeof(float));
        }

        // Element stress (7 components per element: sxx,syy,szz,sxy,syz,sxz,vm)
        int ne7 = num_elements_ * 7;
        if (use_double_) {
            if (static_cast<int>(elem_stress.size()) >= ne7) {
                file_.write(reinterpret_cast<const char*>(elem_stress.data()),
                            ne7 * sizeof(Real));
            } else {
                write_padded_double(elem_stress, ne7);
            }
        } else {
            std::vector<float> fs(ne7, 0.0f);
            for (int i = 0; i < std::min(static_cast<int>(elem_stress.size()), ne7); ++i)
                fs[i] = static_cast<float>(elem_stress[i]);
            file_.write(reinterpret_cast<const char*>(fs.data()), ne7 * sizeof(float));
        }

        num_states_++;
    }

    /// Close the file
    void close() {
        if (!is_open_) return;

        // Write end-of-file marker: time = -999999.0
        Real eof_time = -999999.0;
        if (use_double_) {
            file_.write(reinterpret_cast<const char*>(&eof_time), sizeof(Real));
        } else {
            float feof = static_cast<float>(eof_time);
            file_.write(reinterpret_cast<const char*>(&feof), sizeof(float));
        }

        file_.close();
        is_open_ = false;
    }

    int num_states() const { return num_states_; }
    bool is_open() const { return is_open_; }

private:
    std::ofstream file_;
    std::string filename_;
    int32_t num_nodes_;
    int32_t num_elements_;
    int num_states_;
    bool use_double_;
    bool is_open_;

    void write_padded_double(const std::vector<Real>& data, int expected) {
        file_.write(reinterpret_cast<const char*>(data.data()),
                    data.size() * sizeof(Real));
        int pad = expected - static_cast<int>(data.size());
        if (pad > 0) {
            std::vector<Real> zeros(pad, 0.0);
            file_.write(reinterpret_cast<const char*>(zeros.data()),
                        pad * sizeof(Real));
        }
    }
};

// ############################################################################
// 4. EnSightGoldWriter -- EnSight Gold case/geometry/variable format
// ############################################################################

/**
 * EnSight Gold format consists of:
 *   - Case file (.case): metadata listing geometry + variable files
 *   - Geometry file (.geo): node coords + element connectivity
 *   - Variable files (.scl/.vec/.ten): per-node or per-element data per timestep
 *
 * All binary files use C binary (not Fortran) format: no record markers.
 * Strings are 80-char fixed fields. Integer data is 32-bit.
 * Floating point is 32-bit float in the standard format.
 */
class EnSightGoldWriter {
public:
    static constexpr int DESCRIPTION_LEN = 80;

    EnSightGoldWriter() : is_open_(false), num_nodes_(0) {}

    ~EnSightGoldWriter() {
        if (is_open_) close();
    }

    /// Open: set the base path (without extension)
    bool open(const std::string& base_path) {
        base_path_ = base_path;
        is_open_ = true;
        time_steps_.clear();
        variable_names_.clear();
        variable_files_.clear();
        variable_types_.clear();
        return true;
    }

    /// Write the geometry file (node coordinates and element connectivity)
    /// nodes: flat [x0,y0,z0, ...], connectivity: flat [n0,n1,...,n7, ...] per element
    void write_geometry(const std::vector<Real>& nodes,
                        const std::vector<int32_t>& connectivity,
                        int nodes_per_element = 8) {
        if (!is_open_) return;

        num_nodes_ = static_cast<int>(nodes.size() / 3);
        int num_elems = static_cast<int>(connectivity.size() / nodes_per_element);

        std::string geo_file = base_path_ + ".geo";
        std::ofstream gf(geo_file, std::ios::binary);
        if (!gf.is_open()) return;

        // Description lines (2 x 80 chars)
        write_string80(gf, "EnSight Gold Geometry File");
        write_string80(gf, "Generated by NexusSim");

        // Node ID status and element ID status
        write_string80(gf, "node id off");
        write_string80(gf, "element id off");

        // Part definition
        write_string80(gf, "part");
        int32_t part_id = 1;
        gf.write(reinterpret_cast<const char*>(&part_id), sizeof(part_id));
        write_string80(gf, "NexusSim Part 1");

        // Coordinates
        write_string80(gf, "coordinates");
        int32_t nn32 = static_cast<int32_t>(num_nodes_);
        gf.write(reinterpret_cast<const char*>(&nn32), sizeof(nn32));

        // Write x, y, z as separate float arrays
        std::vector<float> comp(num_nodes_);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < num_nodes_; ++i)
                comp[i] = static_cast<float>(nodes[i * 3 + c]);
            gf.write(reinterpret_cast<const char*>(comp.data()),
                     num_nodes_ * sizeof(float));
        }

        // Element block
        std::string elem_type;
        if (nodes_per_element == 8) elem_type = "hexa8";
        else if (nodes_per_element == 4) elem_type = "tetra4";
        else if (nodes_per_element == 6) elem_type = "penta6";
        else if (nodes_per_element == 3) elem_type = "tria3";
        else elem_type = "hexa8";

        write_string80(gf, elem_type);
        int32_t ne32 = static_cast<int32_t>(num_elems);
        gf.write(reinterpret_cast<const char*>(&ne32), sizeof(ne32));

        // Connectivity (1-based for EnSight)
        std::vector<int32_t> conn_1(connectivity.size());
        for (size_t i = 0; i < connectivity.size(); ++i)
            conn_1[i] = connectivity[i] + 1;
        gf.write(reinterpret_cast<const char*>(conn_1.data()),
                 conn_1.size() * sizeof(int32_t));

        gf.close();
        num_elements_ = num_elems;
        nodes_per_element_ = nodes_per_element;
    }

    /// Write a scalar field variable file
    /// values: one Real per node or per element
    void write_scalar_field(const std::string& name,
                            const std::vector<Real>& values,
                            bool per_element = false,
                            int time_step_index = -1) {
        if (!is_open_) return;

        std::string suffix = (time_step_index >= 0)
            ? "." + zero_pad(time_step_index, 4)
            : "";
        std::string fname = base_path_ + "_" + sanitize(name) + suffix + ".scl";

        std::ofstream vf(fname, std::ios::binary);
        if (!vf.is_open()) return;

        write_string80(vf, name);
        write_string80(vf, "part");
        int32_t pid = 1;
        vf.write(reinterpret_cast<const char*>(&pid), sizeof(pid));

        if (per_element) {
            std::string etype = (nodes_per_element_ == 8) ? "hexa8" :
                                (nodes_per_element_ == 4) ? "tetra4" : "hexa8";
            write_string80(vf, etype);
        } else {
            write_string80(vf, "coordinates");
        }

        int n = static_cast<int>(values.size());
        std::vector<float> fv(n);
        for (int i = 0; i < n; ++i)
            fv[i] = static_cast<float>(values[i]);
        vf.write(reinterpret_cast<const char*>(fv.data()), n * sizeof(float));
        vf.close();

        register_variable(name, "scalar per " + std::string(per_element ? "element" : "node"),
                          sanitize(name));
    }

    /// Write a vector field variable file
    /// values: flat [vx0,vy0,vz0, vx1,vy1,vz1, ...]
    void write_vector_field(const std::string& name,
                            const std::vector<Real>& values,
                            bool per_element = false,
                            int time_step_index = -1) {
        if (!is_open_) return;

        int n = static_cast<int>(values.size() / 3);
        std::string suffix = (time_step_index >= 0)
            ? "." + zero_pad(time_step_index, 4)
            : "";
        std::string fname = base_path_ + "_" + sanitize(name) + suffix + ".vec";

        std::ofstream vf(fname, std::ios::binary);
        if (!vf.is_open()) return;

        write_string80(vf, name);
        write_string80(vf, "part");
        int32_t pid = 1;
        vf.write(reinterpret_cast<const char*>(&pid), sizeof(pid));

        if (per_element) {
            std::string etype = (nodes_per_element_ == 8) ? "hexa8" : "tetra4";
            write_string80(vf, etype);
        } else {
            write_string80(vf, "coordinates");
        }

        // EnSight expects components written as separate arrays: all x, all y, all z
        std::vector<float> comp(n);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < n; ++i)
                comp[i] = static_cast<float>(values[i * 3 + c]);
            vf.write(reinterpret_cast<const char*>(comp.data()), n * sizeof(float));
        }
        vf.close();

        register_variable(name, "vector per " + std::string(per_element ? "element" : "node"),
                          sanitize(name));
    }

    /// Write a symmetric tensor field (6 components in Voigt: xx,yy,zz,xy,yz,xz)
    /// values: flat [s_xx0,s_yy0,...,s_xz0, s_xx1,...]
    void write_tensor_field(const std::string& name,
                            const std::vector<Real>& values,
                            bool per_element = false,
                            int time_step_index = -1) {
        if (!is_open_) return;

        int n = static_cast<int>(values.size() / 6);
        std::string suffix = (time_step_index >= 0)
            ? "." + zero_pad(time_step_index, 4)
            : "";
        std::string fname = base_path_ + "_" + sanitize(name) + suffix + ".ten";

        std::ofstream vf(fname, std::ios::binary);
        if (!vf.is_open()) return;

        write_string80(vf, name);
        write_string80(vf, "part");
        int32_t pid = 1;
        vf.write(reinterpret_cast<const char*>(&pid), sizeof(pid));

        if (per_element) {
            std::string etype = (nodes_per_element_ == 8) ? "hexa8" : "tetra4";
            write_string80(vf, etype);
        } else {
            write_string80(vf, "coordinates");
        }

        // EnSight tensor: 6 components written as separate arrays
        // Order: s11, s22, s33, s12, s23, s13
        std::vector<float> comp(n);
        for (int c = 0; c < 6; ++c) {
            for (int i = 0; i < n; ++i)
                comp[i] = static_cast<float>(values[i * 6 + c]);
            vf.write(reinterpret_cast<const char*>(comp.data()), n * sizeof(float));
        }
        vf.close();

        register_variable(name, "tensor symm per " + std::string(per_element ? "element" : "node"),
                          sanitize(name));
    }

    /// Write the .case file referencing all geometry and variable files
    void write_case_file(const std::vector<Real>& times) {
        if (!is_open_) return;

        time_steps_ = times;
        std::string case_file = base_path_ + ".case";
        std::ofstream cf(case_file);
        if (!cf.is_open()) return;

        // Extract just the basename (no directory prefix)
        std::string base_name = base_path_;
        auto slash_pos = base_name.find_last_of("/\\");
        if (slash_pos != std::string::npos)
            base_name = base_name.substr(slash_pos + 1);

        cf << "FORMAT\n";
        cf << "type: ensight gold\n\n";

        cf << "GEOMETRY\n";
        cf << "model: " << base_name << ".geo\n\n";

        if (!variable_names_.empty()) {
            cf << "VARIABLE\n";
            for (size_t i = 0; i < variable_names_.size(); ++i) {
                cf << variable_types_[i] << ": "
                   << variable_names_[i] << " "
                   << base_name << "_" << variable_files_[i];
                if (!times.empty())
                    cf << ".****";
                std::string ext;
                if (variable_types_[i].find("scalar") != std::string::npos) ext = ".scl";
                else if (variable_types_[i].find("vector") != std::string::npos) ext = ".vec";
                else ext = ".ten";
                cf << ext << "\n";
            }
            cf << "\n";
        }

        if (!times.empty()) {
            cf << "TIME\n";
            cf << "time set: 1\n";
            cf << "number of steps: " << times.size() << "\n";
            cf << "filename start number: 0\n";
            cf << "filename increment: 1\n";
            cf << "time values:\n";
            for (size_t i = 0; i < times.size(); ++i) {
                cf << std::scientific << std::setprecision(6) << times[i] << "\n";
            }
        }

        cf.close();
    }

    /// Close the writer
    void close() {
        is_open_ = false;
    }

    bool is_open() const { return is_open_; }

private:
    std::string base_path_;
    bool is_open_;
    int num_nodes_;
    int num_elements_;
    int nodes_per_element_;
    std::vector<Real> time_steps_;
    std::vector<std::string> variable_names_;
    std::vector<std::string> variable_files_;
    std::vector<std::string> variable_types_;

    void write_string80(std::ofstream& f, const std::string& s) {
        char buf[DESCRIPTION_LEN];
        std::memset(buf, 0, DESCRIPTION_LEN);
        int len = std::min(static_cast<int>(s.size()), DESCRIPTION_LEN - 1);
        std::memcpy(buf, s.c_str(), len);
        f.write(buf, DESCRIPTION_LEN);
    }

    std::string sanitize(const std::string& name) {
        std::string out;
        out.reserve(name.size());
        for (char c : name) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
                out += c;
            else
                out += '_';
        }
        return out;
    }

    std::string zero_pad(int value, int width) {
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(width) << value;
        return oss.str();
    }

    void register_variable(const std::string& name, const std::string& type,
                           const std::string& file_base) {
        // Only register if not already present
        for (size_t i = 0; i < variable_names_.size(); ++i) {
            if (variable_names_[i] == name) return;
        }
        variable_names_.push_back(name);
        variable_types_.push_back(type);
        variable_files_.push_back(file_base);
    }
};

// ############################################################################
// 5. TimeHistoryExporter -- CSV and binary time series at probe locations
// ############################################################################

/**
 * Records nodal or element field values at specific locations over time,
 * then exports to CSV or compact binary format.
 *
 * Probe fields:
 *   - Nodal: displacement_x/y/z, velocity_x/y/z, acceleration_x/y/z
 *   - Element: stress_xx/yy/zz/xy/yz/xz, von_mises, plastic_strain, damage
 *
 * Binary format: [num_probes(int32)][num_steps(int32)]
 *                [probe_headers: id, field_id, per entry]
 *                [time_array(num_steps doubles)]
 *                [value_array(num_probes * num_steps doubles)]
 */
class TimeHistoryExporter {
public:
    enum class ProbeField : int32_t {
        DispX = 0, DispY, DispZ,
        VelX, VelY, VelZ,
        AccX, AccY, AccZ,
        StressXX, StressYY, StressZZ, StressXY, StressYZ, StressXZ,
        VonMises, PlasticStrain, Damage,
        StrainXX, StrainYY, StrainZZ, StrainXY, StrainYZ, StrainXZ
    };

    enum class ProbeLocation : int32_t {
        Node = 0,
        Element = 1
    };

    struct Probe {
        int32_t entity_id;             ///< Node ID or Element ID
        ProbeField field;
        ProbeLocation location;
        std::string label;             ///< Human-readable label
    };

    /// Mesh data snapshot passed to record() for field extraction
    struct MeshSnapshot {
        const Real* node_disp;         ///< [num_nodes * 3], may be nullptr
        const Real* node_vel;          ///< [num_nodes * 3], may be nullptr
        const Real* node_acc;          ///< [num_nodes * 3], may be nullptr
        const Real* elem_stress;       ///< [num_elements * 6] Voigt, may be nullptr
        const Real* elem_strain;       ///< [num_elements * 6] Voigt, may be nullptr
        const Real* elem_plastic;      ///< [num_elements], may be nullptr
        const Real* elem_damage;       ///< [num_elements], may be nullptr
        int num_nodes;
        int num_elements;
    };

    TimeHistoryExporter() = default;

    /// Add a nodal probe
    void add_node_probe(int node_id, ProbeField field,
                        const std::string& label = "") {
        Probe p;
        p.entity_id = node_id;
        p.field = field;
        p.location = ProbeLocation::Node;
        p.label = label.empty() ? ("node_" + std::to_string(node_id) + "_" +
                                    field_name(field)) : label;
        probes_.push_back(p);
    }

    /// Add an element probe
    void add_element_probe(int elem_id, ProbeField field,
                           const std::string& label = "") {
        Probe p;
        p.entity_id = elem_id;
        p.field = field;
        p.location = ProbeLocation::Element;
        p.label = label.empty() ? ("elem_" + std::to_string(elem_id) + "_" +
                                    field_name(field)) : label;
        probes_.push_back(p);
    }

    /// Record a snapshot at the current time
    void record(Real time, const MeshSnapshot& data) {
        times_.push_back(time);

        std::vector<Real> row(probes_.size(), 0.0);
        for (size_t p = 0; p < probes_.size(); ++p) {
            row[p] = extract_value(probes_[p], data);
        }
        values_.push_back(row);
    }

    /// Export all recorded data to CSV
    bool export_csv(const std::string& filename) const {
        std::ofstream f(filename);
        if (!f.is_open()) return false;

        // Header row
        f << "time";
        for (const auto& probe : probes_)
            f << "," << probe.label;
        f << "\n";

        // Data rows
        f << std::scientific << std::setprecision(8);
        for (size_t t = 0; t < times_.size(); ++t) {
            f << times_[t];
            for (size_t p = 0; p < probes_.size(); ++p)
                f << "," << values_[t][p];
            f << "\n";
        }

        f.close();
        return true;
    }

    /// Export to compact binary format
    bool export_binary(const std::string& filename) const {
        std::ofstream f(filename, std::ios::binary);
        if (!f.is_open()) return false;

        int32_t np = static_cast<int32_t>(probes_.size());
        int32_t nt = static_cast<int32_t>(times_.size());

        // Header
        f.write(reinterpret_cast<const char*>(&np), sizeof(np));
        f.write(reinterpret_cast<const char*>(&nt), sizeof(nt));

        // Probe descriptors
        for (const auto& probe : probes_) {
            f.write(reinterpret_cast<const char*>(&probe.entity_id),
                    sizeof(probe.entity_id));
            int32_t fld = static_cast<int32_t>(probe.field);
            f.write(reinterpret_cast<const char*>(&fld), sizeof(fld));
            int32_t loc = static_cast<int32_t>(probe.location);
            f.write(reinterpret_cast<const char*>(&loc), sizeof(loc));
            // Write label (32 chars, null padded)
            char lbl[32];
            std::memset(lbl, 0, 32);
            int len = std::min(static_cast<int>(probe.label.size()), 31);
            std::memcpy(lbl, probe.label.c_str(), len);
            f.write(lbl, 32);
        }

        // Time array
        f.write(reinterpret_cast<const char*>(times_.data()),
                nt * sizeof(Real));

        // Value array (probe-major: all times for probe 0, then probe 1, ...)
        for (int p = 0; p < np; ++p) {
            for (int t = 0; t < nt; ++t) {
                Real v = values_[t][p];
                f.write(reinterpret_cast<const char*>(&v), sizeof(Real));
            }
        }

        f.close();
        return true;
    }

    int num_probes() const { return static_cast<int>(probes_.size()); }
    int num_steps() const { return static_cast<int>(times_.size()); }

    const std::vector<Real>& times() const { return times_; }

    /// Get recorded values for a specific probe index
    std::vector<Real> probe_values(int probe_index) const {
        std::vector<Real> vals(times_.size());
        for (size_t t = 0; t < times_.size(); ++t)
            vals[t] = values_[t][probe_index];
        return vals;
    }

    /// Clear all recorded data (keeps probe definitions)
    void clear_data() {
        times_.clear();
        values_.clear();
    }

    /// Clear everything
    void clear_all() {
        probes_.clear();
        times_.clear();
        values_.clear();
    }

private:
    std::vector<Probe> probes_;
    std::vector<Real> times_;
    std::vector<std::vector<Real>> values_;

    static const char* field_name(ProbeField f) {
        switch (f) {
            case ProbeField::DispX: return "disp_x";
            case ProbeField::DispY: return "disp_y";
            case ProbeField::DispZ: return "disp_z";
            case ProbeField::VelX: return "vel_x";
            case ProbeField::VelY: return "vel_y";
            case ProbeField::VelZ: return "vel_z";
            case ProbeField::AccX: return "acc_x";
            case ProbeField::AccY: return "acc_y";
            case ProbeField::AccZ: return "acc_z";
            case ProbeField::StressXX: return "stress_xx";
            case ProbeField::StressYY: return "stress_yy";
            case ProbeField::StressZZ: return "stress_zz";
            case ProbeField::StressXY: return "stress_xy";
            case ProbeField::StressYZ: return "stress_yz";
            case ProbeField::StressXZ: return "stress_xz";
            case ProbeField::VonMises: return "von_mises";
            case ProbeField::PlasticStrain: return "plastic_strain";
            case ProbeField::Damage: return "damage";
            case ProbeField::StrainXX: return "strain_xx";
            case ProbeField::StrainYY: return "strain_yy";
            case ProbeField::StrainZZ: return "strain_zz";
            case ProbeField::StrainXY: return "strain_xy";
            case ProbeField::StrainYZ: return "strain_yz";
            case ProbeField::StrainXZ: return "strain_xz";
            default: return "unknown";
        }
    }

    /// Extract a single scalar value from the mesh snapshot for a given probe
    Real extract_value(const Probe& probe, const MeshSnapshot& data) const {
        int id = probe.entity_id;

        if (probe.location == ProbeLocation::Node) {
            if (id < 0 || id >= data.num_nodes) return 0.0;

            switch (probe.field) {
                case ProbeField::DispX:
                    return data.node_disp ? data.node_disp[id * 3 + 0] : 0.0;
                case ProbeField::DispY:
                    return data.node_disp ? data.node_disp[id * 3 + 1] : 0.0;
                case ProbeField::DispZ:
                    return data.node_disp ? data.node_disp[id * 3 + 2] : 0.0;
                case ProbeField::VelX:
                    return data.node_vel ? data.node_vel[id * 3 + 0] : 0.0;
                case ProbeField::VelY:
                    return data.node_vel ? data.node_vel[id * 3 + 1] : 0.0;
                case ProbeField::VelZ:
                    return data.node_vel ? data.node_vel[id * 3 + 2] : 0.0;
                case ProbeField::AccX:
                    return data.node_acc ? data.node_acc[id * 3 + 0] : 0.0;
                case ProbeField::AccY:
                    return data.node_acc ? data.node_acc[id * 3 + 1] : 0.0;
                case ProbeField::AccZ:
                    return data.node_acc ? data.node_acc[id * 3 + 2] : 0.0;
                default: return 0.0;
            }
        } else {
            // Element probe
            if (id < 0 || id >= data.num_elements) return 0.0;

            switch (probe.field) {
                case ProbeField::StressXX:
                    return data.elem_stress ? data.elem_stress[id * 6 + 0] : 0.0;
                case ProbeField::StressYY:
                    return data.elem_stress ? data.elem_stress[id * 6 + 1] : 0.0;
                case ProbeField::StressZZ:
                    return data.elem_stress ? data.elem_stress[id * 6 + 2] : 0.0;
                case ProbeField::StressXY:
                    return data.elem_stress ? data.elem_stress[id * 6 + 3] : 0.0;
                case ProbeField::StressYZ:
                    return data.elem_stress ? data.elem_stress[id * 6 + 4] : 0.0;
                case ProbeField::StressXZ:
                    return data.elem_stress ? data.elem_stress[id * 6 + 5] : 0.0;
                case ProbeField::VonMises: {
                    if (!data.elem_stress) return 0.0;
                    const Real* s = &data.elem_stress[id * 6];
                    Real p = (s[0] + s[1] + s[2]) / 3.0;
                    Real dev[6] = {s[0]-p, s[1]-p, s[2]-p, s[3], s[4], s[5]};
                    Real j2 = 0.5*(dev[0]*dev[0] + dev[1]*dev[1] + dev[2]*dev[2])
                              + dev[3]*dev[3] + dev[4]*dev[4] + dev[5]*dev[5];
                    return std::sqrt(3.0 * j2);
                }
                case ProbeField::PlasticStrain:
                    return data.elem_plastic ? data.elem_plastic[id] : 0.0;
                case ProbeField::Damage:
                    return data.elem_damage ? data.elem_damage[id] : 0.0;
                case ProbeField::StrainXX:
                    return data.elem_strain ? data.elem_strain[id * 6 + 0] : 0.0;
                case ProbeField::StrainYY:
                    return data.elem_strain ? data.elem_strain[id * 6 + 1] : 0.0;
                case ProbeField::StrainZZ:
                    return data.elem_strain ? data.elem_strain[id * 6 + 2] : 0.0;
                case ProbeField::StrainXY:
                    return data.elem_strain ? data.elem_strain[id * 6 + 3] : 0.0;
                case ProbeField::StrainYZ:
                    return data.elem_strain ? data.elem_strain[id * 6 + 4] : 0.0;
                case ProbeField::StrainXZ:
                    return data.elem_strain ? data.elem_strain[id * 6 + 5] : 0.0;
                default: return 0.0;
            }
        }
    }
};

// ############################################################################
// 6. CrossSectionForceOutput -- Cutting plane force/moment resultants
// ############################################################################

/**
 * Computes force and moment resultants on user-defined cutting planes.
 * For each cutting plane defined by (point, normal):
 *   F = sum_e (sigma_e . n) * A_e    (traction integration)
 *   M = sum_e (r_e x (sigma_e . n)) * A_e   (moment about the plane point)
 *
 * Elements that straddle the cutting plane are identified by checking if
 * the plane intersects the element's bounding box (centroid-based).
 */
class CrossSectionForceOutput {
public:
    /// Force/moment resultant on a cutting plane
    struct ForceResultant {
        Real Fx, Fy, Fz;
        Real Mx, My, Mz;
        Real time;
        int plane_id;

        ForceResultant()
            : Fx(0), Fy(0), Fz(0), Mx(0), My(0), Mz(0)
            , time(0), plane_id(0) {}

        Real force_magnitude() const {
            return std::sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
        }

        Real moment_magnitude() const {
            return std::sqrt(Mx*Mx + My*My + Mz*Mz);
        }
    };

    /// Cutting plane definition
    struct CuttingPlane {
        int id;
        Real point[3];       ///< A point on the plane
        Real normal[3];      ///< Unit normal to the plane
        std::string name;

        CuttingPlane() : id(0) {
            point[0] = point[1] = point[2] = 0.0;
            normal[0] = 1.0; normal[1] = 0.0; normal[2] = 0.0;
        }
    };

    /// Element geometry data needed for cross-section computation
    struct ElementData {
        Real centroid[3];    ///< Element centroid
        Real area;           ///< Cross-sectional area contribution
        Real stress[6];      ///< Voigt stress: sxx,syy,szz,sxy,syz,sxz
    };

    CrossSectionForceOutput() : next_plane_id_(1) {}

    /// Add a cutting plane
    int add_cutting_plane(const Real point[3], const Real normal[3],
                          const std::string& name = "") {
        CuttingPlane cp;
        cp.id = next_plane_id_++;
        for (int i = 0; i < 3; ++i) {
            cp.point[i] = point[i];
            cp.normal[i] = normal[i];
        }
        // Normalize the normal vector
        Real len = std::sqrt(cp.normal[0]*cp.normal[0] +
                             cp.normal[1]*cp.normal[1] +
                             cp.normal[2]*cp.normal[2]);
        if (len > 1.0e-30) {
            cp.normal[0] /= len;
            cp.normal[1] /= len;
            cp.normal[2] /= len;
        }
        cp.name = name.empty() ? ("plane_" + std::to_string(cp.id)) : name;
        planes_.push_back(cp);
        return cp.id;
    }

    /// Compute force/moment resultants for all cutting planes
    /// elements: all elements with their centroids, areas, and stresses
    /// centroid_tolerance: distance from plane within which an element contributes
    std::vector<ForceResultant> compute_resultants(
            const std::vector<ElementData>& elements,
            Real centroid_tolerance = 1.0e30) const {

        std::vector<ForceResultant> results(planes_.size());

        for (size_t p = 0; p < planes_.size(); ++p) {
            const CuttingPlane& plane = planes_[p];
            ForceResultant& res = results[p];
            res.plane_id = plane.id;

            for (size_t e = 0; e < elements.size(); ++e) {
                const ElementData& elem = elements[e];

                // Compute signed distance from element centroid to plane
                Real dx = elem.centroid[0] - plane.point[0];
                Real dy = elem.centroid[1] - plane.point[1];
                Real dz = elem.centroid[2] - plane.point[2];
                Real dist = dx * plane.normal[0] + dy * plane.normal[1] +
                            dz * plane.normal[2];

                // Only include elements within tolerance of the plane
                if (std::abs(dist) > centroid_tolerance) continue;

                // Compute traction: t = sigma . n
                // sigma is symmetric 3x3 stored as Voigt [sxx,syy,szz,sxy,syz,sxz]
                Real tx = elem.stress[0] * plane.normal[0] +
                          elem.stress[3] * plane.normal[1] +
                          elem.stress[5] * plane.normal[2];
                Real ty = elem.stress[3] * plane.normal[0] +
                          elem.stress[1] * plane.normal[1] +
                          elem.stress[4] * plane.normal[2];
                Real tz = elem.stress[5] * plane.normal[0] +
                          elem.stress[4] * plane.normal[1] +
                          elem.stress[2] * plane.normal[2];

                // Force contribution: F += t * A
                Real A = elem.area;
                res.Fx += tx * A;
                res.Fy += ty * A;
                res.Fz += tz * A;

                // Moment arm: r = centroid - plane.point
                Real rx = elem.centroid[0] - plane.point[0];
                Real ry = elem.centroid[1] - plane.point[1];
                Real rz = elem.centroid[2] - plane.point[2];

                // Moment contribution: M += r x (t * A)
                Real fx = tx * A, fy = ty * A, fz = tz * A;
                res.Mx += ry * fz - rz * fy;
                res.My += rz * fx - rx * fz;
                res.Mz += rx * fy - ry * fx;
            }
        }

        return results;
    }

    /// Convenience: compute for a single plane by ID
    ForceResultant compute_single(int plane_id,
                                  const std::vector<ElementData>& elements,
                                  Real centroid_tolerance = 1.0e30) const {
        for (size_t p = 0; p < planes_.size(); ++p) {
            if (planes_[p].id == plane_id) {
                auto all = compute_resultants(elements, centroid_tolerance);
                return all[p];
            }
        }
        return ForceResultant();
    }

    /// Record time-stamped resultants for later export
    void record(Real time, const std::vector<ElementData>& elements,
                Real centroid_tolerance = 1.0e30) {
        auto results = compute_resultants(elements, centroid_tolerance);
        for (auto& r : results) r.time = time;
        history_.insert(history_.end(), results.begin(), results.end());
    }

    /// Export recorded history to CSV
    bool export_csv(const std::string& filename) const {
        std::ofstream f(filename);
        if (!f.is_open()) return false;

        f << "time,plane_id,plane_name,Fx,Fy,Fz,Mx,My,Mz,F_mag,M_mag\n";
        f << std::scientific << std::setprecision(8);

        for (const auto& r : history_) {
            std::string pname;
            for (const auto& plane : planes_) {
                if (plane.id == r.plane_id) { pname = plane.name; break; }
            }
            f << r.time << "," << r.plane_id << "," << pname << ","
              << r.Fx << "," << r.Fy << "," << r.Fz << ","
              << r.Mx << "," << r.My << "," << r.Mz << ","
              << r.force_magnitude() << "," << r.moment_magnitude() << "\n";
        }

        f.close();
        return true;
    }

    int num_planes() const { return static_cast<int>(planes_.size()); }
    const std::vector<CuttingPlane>& planes() const { return planes_; }
    const std::vector<ForceResultant>& history() const { return history_; }

    void clear_history() { history_.clear(); }

private:
    std::vector<CuttingPlane> planes_;
    std::vector<ForceResultant> history_;
    int next_plane_id_;
};

} // namespace io
} // namespace nxs
