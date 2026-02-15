#pragma once

/**
 * @file extended_checkpoint.hpp
 * @brief Extended checkpoint with failure, rigid body, contact, and tied states
 *
 * Composes the existing CheckpointWriter/Reader to append typed sections
 * after material state data. Each section has a header (type, size, count)
 * followed by fixed-size records. Unknown sections are skippable.
 *
 * Extended sections:
 *   1. Failure states (damage, failed flag, failure mode, history[16])
 *   2. Rigid body states (COM, velocity, angular velocity, quaternion, force, torque)
 *   3. Contact states (gap, normal, tangent slip, active/sticking flags)
 *   4. Tied pair states (slave/master nodes, weights, failure tracking)
 *   5. Rigid wall states (id, origin, velocity)
 */

#include <nexussim/io/checkpoint.hpp>
#include <nexussim/physics/failure/failure_model.hpp>
#include <nexussim/fem/rigid_body.hpp>
#include <nexussim/fem/contact.hpp>
#include <nexussim/fem/tied_contact.hpp>
#include <nexussim/fem/rigid_wall.hpp>

namespace nxs {
namespace io {

// ============================================================================
// Section Types
// ============================================================================

enum class ExtendedSectionType : uint32_t {
    FailureState   = 1,
    RigidBody      = 2,
    ContactState   = 3,
    TiedPair       = 4,
    RigidWallState = 5
};

// ============================================================================
// Extended Section Header
// ============================================================================

struct ExtendedSectionHeader {
    uint32_t section_type;      ///< ExtendedSectionType as uint32
    uint32_t reserved;          ///< Padding for alignment
    uint64_t section_size_bytes;///< Total bytes of records (excluding this header)
    uint64_t num_entries;       ///< Number of records

    ExtendedSectionHeader()
        : section_type(0), reserved(0), section_size_bytes(0), num_entries(0) {}
};

static_assert(sizeof(ExtendedSectionHeader) == 24, "Section header must be 24 bytes");

// ============================================================================
// Serializable Data Structs (fixed-size for binary I/O)
// ============================================================================

struct FailureStateData {
    Real damage;
    int32_t failed;
    int32_t failure_mode;
    Real history[16];

    FailureStateData() : damage(0.0), failed(0), failure_mode(0) {
        for (int i = 0; i < 16; ++i) history[i] = 0.0;
    }

    void from_failure_state(const physics::failure::FailureState& fs) {
        damage = fs.damage;
        failed = fs.failed ? 1 : 0;
        failure_mode = fs.failure_mode;
        for (int i = 0; i < 16; ++i) history[i] = fs.history[i];
    }

    void to_failure_state(physics::failure::FailureState& fs) const {
        fs.damage = damage;
        fs.failed = (failed != 0);
        fs.failure_mode = failure_mode;
        for (int i = 0; i < 16; ++i) fs.history[i] = history[i];
    }
};

struct RigidBodyStateData {
    int32_t id;
    int32_t reserved;
    Real com[3];
    Real velocity[3];
    Real angular_velocity[3];
    Real orientation[4];   // quaternion: w, x, y, z
    Real force[3];
    Real torque[3];

    RigidBodyStateData() : id(0), reserved(0) {
        for (int i = 0; i < 3; ++i) {
            com[i] = 0.0; velocity[i] = 0.0;
            angular_velocity[i] = 0.0; force[i] = 0.0; torque[i] = 0.0;
        }
        orientation[0] = 1.0; orientation[1] = 0.0;
        orientation[2] = 0.0; orientation[3] = 0.0;
    }

    void from_rigid_body(const fem::RigidBody& rb) {
        id = rb.id();
        const auto& props = rb.properties();
        for (int i = 0; i < 3; ++i) {
            com[i] = props.com[i];
            velocity[i] = rb.velocity()[i];
            angular_velocity[i] = rb.angular_velocity()[i];
            force[i] = rb.force()[i];
            torque[i] = rb.torque()[i];
        }
        const auto& q = rb.orientation();
        orientation[0] = q.w;
        orientation[1] = q.x;
        orientation[2] = q.y;
        orientation[3] = q.z;
    }

    void to_rigid_body(fem::RigidBody& rb) const {
        auto& props = rb.properties();
        for (int i = 0; i < 3; ++i) {
            props.com[i] = com[i];
            rb.velocity()[i] = velocity[i];
            rb.angular_velocity()[i] = angular_velocity[i];
            rb.force()[i] = force[i];
            rb.torque()[i] = torque[i];
        }
        // Note: quaternion is read-only in RigidBody (via orientation())
        // so we store it for verification but can't set it back directly
    }
};

struct ContactStateData {
    uint64_t slave_node;
    uint64_t master_segment;
    Real gap;
    Real normal[3];
    Real tangent_slip[3];
    int32_t active;
    int32_t sticking;

    ContactStateData()
        : slave_node(0), master_segment(0), gap(0.0)
        , active(0), sticking(0) {
        for (int i = 0; i < 3; ++i) { normal[i] = 0.0; tangent_slip[i] = 0.0; }
    }

    void from_contact_pair(const fem::ContactPair& cp, bool is_active = true, bool is_sticking = false) {
        slave_node = cp.slave_node;
        master_segment = cp.master_face;
        gap = cp.penetration_depth;
        for (int i = 0; i < 3; ++i) normal[i] = cp.normal[i];
        // tangent_slip populated from external friction state
        active = is_active ? 1 : 0;
        sticking = is_sticking ? 1 : 0;
    }

    void to_contact_pair(fem::ContactPair& cp, bool& is_active, bool& is_sticking) const {
        cp.slave_node = slave_node;
        cp.master_face = master_segment;
        cp.penetration_depth = gap;
        for (int i = 0; i < 3; ++i) cp.normal[i] = normal[i];
        is_active = (active != 0);
        is_sticking = (sticking != 0);
    }
};

struct TiedPairData {
    uint64_t slave_node;
    uint64_t master_nodes[4];
    Real phi[4];
    int32_t num_master_nodes;
    int32_t active;
    Real accumulated_force;
    Real max_force;
    Real gap_initial[3];

    TiedPairData()
        : slave_node(0), num_master_nodes(0), active(1)
        , accumulated_force(0.0), max_force(0.0) {
        for (int i = 0; i < 4; ++i) { master_nodes[i] = 0; phi[i] = 0.0; }
        gap_initial[0] = gap_initial[1] = gap_initial[2] = 0.0;
    }

    void from_tied_pair(const fem::TiedPair& tp) {
        slave_node = tp.slave_node;
        for (int i = 0; i < 4; ++i) {
            master_nodes[i] = tp.master_nodes[i];
            phi[i] = tp.phi[i];
        }
        num_master_nodes = tp.num_master_nodes;
        active = tp.active ? 1 : 0;
        accumulated_force = tp.accumulated_force;
        max_force = tp.max_force;
        for (int i = 0; i < 3; ++i) gap_initial[i] = tp.gap_initial[i];
    }

    void to_tied_pair(fem::TiedPair& tp) const {
        tp.slave_node = slave_node;
        for (int i = 0; i < 4; ++i) {
            tp.master_nodes[i] = master_nodes[i];
            tp.phi[i] = phi[i];
        }
        tp.num_master_nodes = num_master_nodes;
        tp.active = (active != 0);
        tp.accumulated_force = accumulated_force;
        tp.max_force = max_force;
        for (int i = 0; i < 3; ++i) tp.gap_initial[i] = gap_initial[i];
    }
};

struct RigidWallStateData {
    int32_t id;
    int32_t reserved;
    Real origin[3];
    Real velocity[3];

    RigidWallStateData() : id(0), reserved(0) {
        for (int i = 0; i < 3; ++i) { origin[i] = 0.0; velocity[i] = 0.0; }
    }

    void from_wall_config(const fem::RigidWallConfig& wc) {
        id = wc.id;
        for (int i = 0; i < 3; ++i) {
            origin[i] = wc.origin[i];
            velocity[i] = wc.velocity[i];
        }
    }

    void to_wall_config(fem::RigidWallConfig& wc) const {
        wc.id = id;
        for (int i = 0; i < 3; ++i) {
            wc.origin[i] = origin[i];
            wc.velocity[i] = velocity[i];
        }
    }
};

// ============================================================================
// Extended Checkpoint Writer
// ============================================================================

class ExtendedCheckpointWriter {
public:
    explicit ExtendedCheckpointWriter(const std::string& filename)
        : filename_(filename) {}

    bool write(const Mesh& mesh,
               const State& state,
               const std::vector<physics::MaterialState>& material_states = {},
               const std::vector<physics::failure::FailureState>& failure_states = {},
               const std::vector<fem::RigidBody*>& rigid_bodies = {},
               const std::vector<fem::ContactPair>& contact_pairs = {},
               const std::vector<fem::TiedPair>& tied_pairs = {},
               const std::vector<fem::RigidWallConfig>& wall_configs = {}) {

        // First write standard checkpoint data
        std::ofstream file(filename_, std::ios::binary);
        if (!file.is_open()) return false;

        // Build and write header (standard v1 format)
        CheckpointHeader header;
        header.num_nodes = mesh.num_nodes();
        header.num_elements = mesh.num_elements();
        header.num_fields = static_cast<uint32_t>(state.field_names().size());
        header.num_material_states = material_states.size();
        header.time = state.time();
        header.step = state.step();

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write mesh coordinates
        write_mesh_coordinates(file, mesh);

        // Write state fields
        auto names = state.field_names();
        for (const auto& name : names) {
            write_field(file, name, state.field(name));
        }

        // Write material states
        if (!material_states.empty()) {
            std::vector<MaterialStateData> data(material_states.size());
            for (std::size_t i = 0; i < material_states.size(); ++i) {
                data[i].from_material_state(material_states[i]);
            }
            file.write(reinterpret_cast<const char*>(data.data()),
                       static_cast<std::streamsize>(data.size() * sizeof(MaterialStateData)));
        }

        // --- Extended sections ---

        // Failure states
        if (!failure_states.empty()) {
            write_section(file, ExtendedSectionType::FailureState,
                          failure_states.size(), [&](std::ofstream& f) {
                std::vector<FailureStateData> fsd(failure_states.size());
                for (std::size_t i = 0; i < failure_states.size(); ++i) {
                    fsd[i].from_failure_state(failure_states[i]);
                }
                f.write(reinterpret_cast<const char*>(fsd.data()),
                        static_cast<std::streamsize>(fsd.size() * sizeof(FailureStateData)));
            });
        }

        // Rigid bodies
        if (!rigid_bodies.empty()) {
            write_section(file, ExtendedSectionType::RigidBody,
                          rigid_bodies.size(), [&](std::ofstream& f) {
                std::vector<RigidBodyStateData> rbd(rigid_bodies.size());
                for (std::size_t i = 0; i < rigid_bodies.size(); ++i) {
                    rbd[i].from_rigid_body(*rigid_bodies[i]);
                }
                f.write(reinterpret_cast<const char*>(rbd.data()),
                        static_cast<std::streamsize>(rbd.size() * sizeof(RigidBodyStateData)));
            });
        }

        // Contact states
        if (!contact_pairs.empty()) {
            write_section(file, ExtendedSectionType::ContactState,
                          contact_pairs.size(), [&](std::ofstream& f) {
                std::vector<ContactStateData> csd(contact_pairs.size());
                for (std::size_t i = 0; i < contact_pairs.size(); ++i) {
                    csd[i].from_contact_pair(contact_pairs[i]);
                }
                f.write(reinterpret_cast<const char*>(csd.data()),
                        static_cast<std::streamsize>(csd.size() * sizeof(ContactStateData)));
            });
        }

        // Tied pairs
        if (!tied_pairs.empty()) {
            write_section(file, ExtendedSectionType::TiedPair,
                          tied_pairs.size(), [&](std::ofstream& f) {
                std::vector<TiedPairData> tpd(tied_pairs.size());
                for (std::size_t i = 0; i < tied_pairs.size(); ++i) {
                    tpd[i].from_tied_pair(tied_pairs[i]);
                }
                f.write(reinterpret_cast<const char*>(tpd.data()),
                        static_cast<std::streamsize>(tpd.size() * sizeof(TiedPairData)));
            });
        }

        // Rigid wall states
        if (!wall_configs.empty()) {
            write_section(file, ExtendedSectionType::RigidWallState,
                          wall_configs.size(), [&](std::ofstream& f) {
                std::vector<RigidWallStateData> wsd(wall_configs.size());
                for (std::size_t i = 0; i < wall_configs.size(); ++i) {
                    wsd[i].from_wall_config(wall_configs[i]);
                }
                f.write(reinterpret_cast<const char*>(wsd.data()),
                        static_cast<std::streamsize>(wsd.size() * sizeof(RigidWallStateData)));
            });
        }

        file.flush();
        bytes_written_ = file.tellp();
        return file.good();
    }

    std::size_t bytes_written() const { return bytes_written_; }

private:
    template<typename WriteFunc>
    void write_section(std::ofstream& file, ExtendedSectionType type,
                       std::size_t count, WriteFunc func) {
        ExtendedSectionHeader shdr;
        shdr.section_type = static_cast<uint32_t>(type);
        shdr.num_entries = count;

        // Write header placeholder, remember position
        auto hdr_pos = file.tellp();
        file.write(reinterpret_cast<const char*>(&shdr), sizeof(shdr));

        auto data_start = file.tellp();
        func(file);
        auto data_end = file.tellp();

        // Go back and fill in section_size_bytes
        shdr.section_size_bytes = static_cast<uint64_t>(data_end - data_start);
        file.seekp(hdr_pos);
        file.write(reinterpret_cast<const char*>(&shdr), sizeof(shdr));
        file.seekp(data_end);
    }

    void write_mesh_coordinates(std::ofstream& file, const Mesh& mesh) {
        std::size_t n = mesh.num_nodes();
        const auto& coords = mesh.coordinates();
        for (std::size_t comp = 0; comp < 3; ++comp) {
            auto span = coords.component(comp);
            file.write(reinterpret_cast<const char*>(span.data()),
                       static_cast<std::streamsize>(n * sizeof(Real)));
        }
    }

    void write_field(std::ofstream& file, const std::string& name,
                     const Field<Real>& field) {
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);

        uint32_t ftype = static_cast<uint32_t>(field.type());
        uint32_t floc = static_cast<uint32_t>(field.location());
        uint64_t nentities = field.num_entities();
        uint32_t ncomp = static_cast<uint32_t>(field.num_components());

        file.write(reinterpret_cast<const char*>(&ftype), sizeof(ftype));
        file.write(reinterpret_cast<const char*>(&floc), sizeof(floc));
        file.write(reinterpret_cast<const char*>(&nentities), sizeof(nentities));
        file.write(reinterpret_cast<const char*>(&ncomp), sizeof(ncomp));
        file.write(reinterpret_cast<const char*>(field.data()),
                   static_cast<std::streamsize>(nentities * ncomp * sizeof(Real)));
    }

    std::string filename_;
    std::size_t bytes_written_ = 0;
};

// ============================================================================
// Extended Checkpoint Reader
// ============================================================================

class ExtendedCheckpointReader {
public:
    explicit ExtendedCheckpointReader(const std::string& filename)
        : filename_(filename) {}

    bool read(Mesh& mesh, State& state,
              std::vector<physics::MaterialState>* material_states = nullptr,
              std::vector<physics::failure::FailureState>* failure_states = nullptr,
              std::vector<RigidBodyStateData>* rigid_body_states = nullptr,
              std::vector<ContactStateData>* contact_states = nullptr,
              std::vector<fem::TiedPair>* tied_pairs = nullptr,
              std::vector<fem::RigidWallConfig>* wall_configs = nullptr) {

        std::ifstream file(filename_, std::ios::binary);
        if (!file.is_open()) return false;

        // Read standard header
        file.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!file.good() || !header_.is_valid()) return false;
        header_valid_ = true;

        state.set_time(header_.time);
        state.set_step(header_.step);

        // Read mesh coordinates
        if (header_.num_nodes > 0) {
            std::size_t n = header_.num_nodes;
            auto& coords = mesh.coordinates();
            for (std::size_t comp = 0; comp < 3; ++comp) {
                auto span = coords.component(comp);
                file.read(reinterpret_cast<char*>(span.data()),
                          static_cast<std::streamsize>(n * sizeof(Real)));
            }
        }

        // Read fields
        for (uint32_t i = 0; i < header_.num_fields; ++i) {
            read_field(file, state);
        }

        // Read material states
        if (header_.num_material_states > 0) {
            std::size_t n = header_.num_material_states;
            std::vector<MaterialStateData> data(n);
            file.read(reinterpret_cast<char*>(data.data()),
                      static_cast<std::streamsize>(n * sizeof(MaterialStateData)));

            if (material_states) {
                material_states->resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    data[i].to_material_state((*material_states)[i]);
                }
            }
        }

        // Reset section tracking
        has_failure_ = false;
        has_rigid_body_ = false;
        has_contact_ = false;
        has_tied_ = false;
        has_wall_ = false;

        // Read extended sections (if any remain in file)
        while (file.good() && file.peek() != EOF) {
            ExtendedSectionHeader shdr;
            file.read(reinterpret_cast<char*>(&shdr), sizeof(shdr));
            if (!file.good()) break;

            auto section_type = static_cast<ExtendedSectionType>(shdr.section_type);

            switch (section_type) {
                case ExtendedSectionType::FailureState: {
                    has_failure_ = true;
                    if (failure_states) {
                        std::vector<FailureStateData> fsd(shdr.num_entries);
                        file.read(reinterpret_cast<char*>(fsd.data()),
                                  static_cast<std::streamsize>(shdr.section_size_bytes));
                        failure_states->resize(shdr.num_entries);
                        for (std::size_t i = 0; i < shdr.num_entries; ++i) {
                            fsd[i].to_failure_state((*failure_states)[i]);
                        }
                    } else {
                        file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                                   std::ios::cur);
                    }
                    break;
                }
                case ExtendedSectionType::RigidBody: {
                    has_rigid_body_ = true;
                    if (rigid_body_states) {
                        rigid_body_states->resize(shdr.num_entries);
                        file.read(reinterpret_cast<char*>(rigid_body_states->data()),
                                  static_cast<std::streamsize>(shdr.section_size_bytes));
                    } else {
                        file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                                   std::ios::cur);
                    }
                    break;
                }
                case ExtendedSectionType::ContactState: {
                    has_contact_ = true;
                    if (contact_states) {
                        contact_states->resize(shdr.num_entries);
                        file.read(reinterpret_cast<char*>(contact_states->data()),
                                  static_cast<std::streamsize>(shdr.section_size_bytes));
                    } else {
                        file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                                   std::ios::cur);
                    }
                    break;
                }
                case ExtendedSectionType::TiedPair: {
                    has_tied_ = true;
                    if (tied_pairs) {
                        std::vector<TiedPairData> tpd(shdr.num_entries);
                        file.read(reinterpret_cast<char*>(tpd.data()),
                                  static_cast<std::streamsize>(shdr.section_size_bytes));
                        tied_pairs->resize(shdr.num_entries);
                        for (std::size_t i = 0; i < shdr.num_entries; ++i) {
                            tpd[i].to_tied_pair((*tied_pairs)[i]);
                        }
                    } else {
                        file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                                   std::ios::cur);
                    }
                    break;
                }
                case ExtendedSectionType::RigidWallState: {
                    has_wall_ = true;
                    if (wall_configs) {
                        std::vector<RigidWallStateData> wsd(shdr.num_entries);
                        file.read(reinterpret_cast<char*>(wsd.data()),
                                  static_cast<std::streamsize>(shdr.section_size_bytes));
                        wall_configs->resize(shdr.num_entries);
                        for (std::size_t i = 0; i < shdr.num_entries; ++i) {
                            wsd[i].to_wall_config((*wall_configs)[i]);
                        }
                    } else {
                        file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                                   std::ios::cur);
                    }
                    break;
                }
                default:
                    // Unknown section — skip it
                    file.seekg(static_cast<std::streamoff>(shdr.section_size_bytes),
                               std::ios::cur);
                    break;
            }
        }

        return true;
    }

    bool has_section(ExtendedSectionType type) const {
        switch (type) {
            case ExtendedSectionType::FailureState:   return has_failure_;
            case ExtendedSectionType::RigidBody:      return has_rigid_body_;
            case ExtendedSectionType::ContactState:    return has_contact_;
            case ExtendedSectionType::TiedPair:        return has_tied_;
            case ExtendedSectionType::RigidWallState:  return has_wall_;
            default: return false;
        }
    }

    const CheckpointHeader& header() const { return header_; }
    bool header_valid() const { return header_valid_; }
    Real time() const { return header_.time; }
    int64_t step() const { return header_.step; }

private:
    void read_field(std::ifstream& file, State& state) {
        uint32_t name_len = 0;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        uint32_t ftype, floc, ncomp;
        uint64_t nentities;
        file.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
        file.read(reinterpret_cast<char*>(&floc), sizeof(floc));
        file.read(reinterpret_cast<char*>(&nentities), sizeof(nentities));
        file.read(reinterpret_cast<char*>(&ncomp), sizeof(ncomp));

        std::size_t data_bytes = nentities * ncomp * sizeof(Real);

        if (state.has_field(name)) {
            auto& existing = state.field(name);
            file.read(reinterpret_cast<char*>(existing.data()),
                      static_cast<std::streamsize>(data_bytes));
        } else {
            Field<Real> new_field(name,
                                   static_cast<FieldType>(ftype),
                                   static_cast<FieldLocation>(floc),
                                   nentities, ncomp);
            file.read(reinterpret_cast<char*>(new_field.data()),
                      static_cast<std::streamsize>(data_bytes));
            state.add_field(name, std::move(new_field));
        }
    }

    std::string filename_;
    CheckpointHeader header_;
    bool header_valid_ = false;
    bool has_failure_ = false;
    bool has_rigid_body_ = false;
    bool has_contact_ = false;
    bool has_tied_ = false;
    bool has_wall_ = false;
};

} // namespace io
} // namespace nxs
