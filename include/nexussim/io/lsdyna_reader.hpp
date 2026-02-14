#pragma once

/**
 * @file lsdyna_reader.hpp
 * @brief LS-DYNA keyword format reader
 *
 * Parses LS-DYNA keyword (.k, .dyn, .key) format files and creates NexusSim mesh objects.
 *
 * Supported keywords:
 * - *NODE - Node definitions
 * - *ELEMENT_SOLID/SHELL/BEAM - Element definitions
 * - *PART - Part definitions
 * - *MAT_ELASTIC (MAT_001), *MAT_ORTHOTROPIC_ELASTIC (MAT_002),
 *   *MAT_PLASTIC_KINEMATIC (MAT_003), *MAT_VISCOELASTIC (MAT_006),
 *   *MAT_NULL (MAT_009), *MAT_ELASTIC_PLASTIC_HYDRO (MAT_010),
 *   *MAT_JOHNSON_COOK (MAT_015), *MAT_RIGID (MAT_020),
 *   *MAT_PIECEWISE_LINEAR_PLASTICITY (MAT_024), *MAT_HONEYCOMB (MAT_026),
 *   *MAT_MOONEY-RIVLIN_RUBBER (MAT_027), *MAT_FOAM (MAT_057),
 *   *MAT_CRUSHABLE_FOAM (MAT_063), *MAT_OGDEN_RUBBER (MAT_077),
 *   *MAT_COWPER_SYMONDS
 * - *SECTION_SHELL, *SECTION_SOLID, *SECTION_BEAM
 * - *SET_NODE_LIST, *SET_PART_LIST, *SET_SHELL_LIST, *SET_SOLID_LIST
 * - *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE, *CONTACT_AUTOMATIC_NODES_TO_SURFACE,
 *   *CONTACT_AUTOMATIC_SINGLE_SURFACE, *CONTACT_TIED_SURFACE_TO_SURFACE
 * - *EOS_LINEAR_POLYNOMIAL, *EOS_GRUNEISEN, *EOS_JWL
 * - *INITIAL_VELOCITY, *INITIAL_VELOCITY_GENERATION
 * - *CONTROL_TERMINATION, *CONTROL_TIMESTEP
 * - *LOAD_BODY_X/Y/Z, *LOAD_NODE_SET, *BOUNDARY_SPC_SET,
 *   *BOUNDARY_PRESCRIBED_MOTION_SET
 * - *DEFINE_CURVE, *DATABASE_BINARY_D3PLOT
 * - *TITLE, *END
 *
 * File format reference: LS-DYNA Keyword User's Manual
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/section.hpp>
#include <nexussim/physics/eos.hpp>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <algorithm>
#include <cctype>

namespace nxs {
namespace io {

// Forward declarations
struct LSDynaNode;
struct LSDynaElement;
struct LSDynaMaterial;
struct LSDynaPart;
struct LSDynaNodeSet;
struct LSDynaSPC;
struct LSDynaLoad;
struct LSDynaSection;
struct LSDynaContact;
struct LSDynaEOS;
struct LSDynaPartSet;
struct LSDynaElementSet;
struct LSDynaInitVelocity;
struct LSDynaControl;
struct LSDynaBodyLoad;
struct LSDynaPrescribedMotion;

/**
 * @brief LS-DYNA node data
 */
struct LSDynaNode {
    Index nid;      // Node ID
    Real x, y, z;   // Coordinates
    Int tc;         // Translational constraint code
    Int rc;         // Rotational constraint code
};

/**
 * @brief LS-DYNA element data
 */
struct LSDynaElement {
    Index eid;              // Element ID
    Index pid;              // Part ID
    ElementType type;       // NexusSim element type
    std::vector<Index> nodes;  // Node IDs
};

/**
 * @brief LS-DYNA material data
 */
struct LSDynaMaterial {
    Index mid;              // Material ID
    std::string type;       // Material type string (e.g., "ELASTIC")
    std::string title;      // Optional title

    // Common properties
    Real ro;                // Density
    Real e;                 // Young's modulus
    Real pr;                // Poisson's ratio

    // Johnson-Cook parameters
    Real a_jc;              // Yield stress A
    Real b_jc;              // Hardening B
    Real n_jc;              // Hardening exponent n
    Real c_jc;              // Strain rate coefficient C
    Real m_jc;              // Thermal softening m
    Real tm;                // Melting temperature
    Real tr;                // Room temperature
    Real epso;              // Reference strain rate

    // Piecewise linear plasticity
    Real sigy;              // Initial yield stress
    Real etan;              // Tangent modulus
    Real fail;              // Failure strain
    Index lcss;             // Load curve ID for stress-strain

    // Rigid body parameters
    Real cmo;               // Center-of-mass constraint
    Real con1, con2;        // Constraint directions

    // Shear modulus / viscosity
    Real mu;

    // Mooney-Rivlin
    Real c10_mr, c01_mr;

    // Ogden
    Real mu1, mu2, mu3, alpha1, alpha2, alpha3;

    // Orthotropic
    Real ea, eb, ec, gab, gbc, gca, prba, prca, prcb;

    // Foam/crushable foam
    Real damp_foam, e_crush, densification, tension_cutoff;

    // Cowper-Symonds
    Real cs_d, cs_q;

    // Viscoelastic
    Real bulk, g0, gi, beta_ve;

    // Hydro
    Real g_shear;

    // Honeycomb
    Real vf, bulk_hc;

    LSDynaMaterial()
        : mid(0), ro(0), e(0), pr(0)
        , a_jc(0), b_jc(0), n_jc(0), c_jc(0), m_jc(0)
        , tm(0), tr(0), epso(1.0)
        , sigy(0), etan(0), fail(0), lcss(0)
        , cmo(0), con1(0), con2(0)
        , mu(0)
        , c10_mr(0), c01_mr(0)
        , mu1(0), mu2(0), mu3(0), alpha1(0), alpha2(0), alpha3(0)
        , ea(0), eb(0), ec(0), gab(0), gbc(0), gca(0), prba(0), prca(0), prcb(0)
        , damp_foam(0), e_crush(0), densification(0), tension_cutoff(0)
        , cs_d(0), cs_q(0)
        , bulk(0), g0(0), gi(0), beta_ve(0)
        , g_shear(0)
        , vf(0), bulk_hc(0)
    {}
};

/**
 * @brief LS-DYNA part data
 */
struct LSDynaPart {
    Index pid;          // Part ID
    Index secid;        // Section ID
    Index mid;          // Material ID
    std::string title;  // Part title
};

/**
 * @brief LS-DYNA node set data
 */
struct LSDynaNodeSet {
    Index sid;                  // Set ID
    std::string title;          // Set title
    std::vector<Index> nodes;   // Node IDs in set
};

/**
 * @brief LS-DYNA single point constraint
 */
struct LSDynaSPC {
    Index nsid;         // Node set ID
    Int cid;            // Coordinate system ID
    Int dofx, dofy, dofz;       // DOF constraints (0=free, 1=fixed)
    Int dofrx, dofry, dofrz;    // Rotational DOF constraints

    LSDynaSPC()
        : nsid(0), cid(0)
        , dofx(0), dofy(0), dofz(0)
        , dofrx(0), dofry(0), dofrz(0)
    {}
};

/**
 * @brief LS-DYNA nodal load
 */
struct LSDynaLoad {
    Index nsid;         // Node set ID
    Int dof;            // Degree of freedom (1=x, 2=y, 3=z)
    Index lcid;         // Load curve ID
    Real sf;            // Scale factor

    LSDynaLoad() : nsid(0), dof(0), lcid(0), sf(1.0) {}
};

/**
 * @brief LS-DYNA load curve
 */
struct LSDynaLoadCurve {
    Index lcid;                         // Load curve ID
    std::vector<std::pair<Real, Real>> points;  // (time, value) pairs
};

/**
 * @brief LS-DYNA section data
 */
struct LSDynaSection {
    Index secid;
    std::string type;   // "SHELL", "SOLID", "BEAM"
    int elform;
    Real shrf;          // Shear correction factor
    int nip;            // Through-thickness integration points
    Real t1, t2, t3, t4; // Nodal thicknesses
    int cst;            // Cross section type (beam)
    Real ts1, ts2;      // Beam dimensions
    Real area, iss, ist, j_torsion;

    LSDynaSection() : secid(0), elform(2), shrf(1.0), nip(2),
        t1(0), t2(0), t3(0), t4(0), cst(0), ts1(0), ts2(0),
        area(0), iss(0), ist(0), j_torsion(0) {}
};

/**
 * @brief LS-DYNA contact data
 */
struct LSDynaContact {
    Index id;
    std::string type;       // "SURFACE_TO_SURFACE", "NODES_TO_SURFACE", "SINGLE_SURFACE", "TIED"
    Index ssid;             // Slave set/part ID
    Index msid;             // Master set/part ID
    int sstyp;              // Slave type
    int mstyp;              // Master type
    Real fs, fd;            // Static/dynamic friction
    Real dc;                // Exponential decay
    Real vc;                // Viscous friction coefficient
    Real sfs, sfm;          // Penalty scale factors
    Real soft;              // Soft constraint option

    LSDynaContact() : id(0), ssid(0), msid(0), sstyp(0), mstyp(0),
        fs(0), fd(0), dc(0), vc(0), sfs(1.0), sfm(1.0), soft(0) {}
};

/**
 * @brief LS-DYNA equation of state data
 */
struct LSDynaEOS {
    Index eosid;
    std::string type;       // "GRUNEISEN", "JWL", "LINEAR_POLYNOMIAL"
    // Gruneisen
    Real C0, S1, S2, S3, gamma0, a_coeff, E0, V0;
    // JWL
    Real A_jwl, B_jwl, R1, R2, omega, E0_jwl, V0_jwl;
    // Polynomial
    Real cpoly[7]; // C0..C6

    LSDynaEOS() : eosid(0), C0(0), S1(0), S2(0), S3(0), gamma0(0),
        a_coeff(0), E0(0), V0(1.0), A_jwl(0), B_jwl(0), R1(0), R2(0),
        omega(0), E0_jwl(0), V0_jwl(1.0) {
        for (int i = 0; i < 7; i++) cpoly[i] = 0;
    }
};

/**
 * @brief LS-DYNA part set data
 */
struct LSDynaPartSet {
    Index sid;
    std::vector<Index> parts;
};

/**
 * @brief LS-DYNA element set data
 */
struct LSDynaElementSet {
    Index sid;
    std::vector<Index> elements;
};

/**
 * @brief LS-DYNA initial velocity data
 */
struct LSDynaInitVelocity {
    Index nsid;         // Node set ID (0 = all)
    Real vx, vy, vz;   // Translational velocity
    Real vrx, vry, vrz; // Rotational velocity

    LSDynaInitVelocity() : nsid(0), vx(0), vy(0), vz(0), vrx(0), vry(0), vrz(0) {}
};

/**
 * @brief LS-DYNA control parameters
 */
struct LSDynaControl {
    Real termination_time;
    Real dt_init;       // Initial timestep
    Real dt_scale;      // Timestep scale factor (TSSFAC)
    Real dt_min;        // Minimum timestep
    int dt_ms_flag;     // Mass scaling flag

    LSDynaControl() : termination_time(1.0), dt_init(0.0),
        dt_scale(0.9), dt_min(0.0), dt_ms_flag(0) {}
};

/**
 * @brief LS-DYNA body load data
 */
struct LSDynaBodyLoad {
    int dof;            // Direction (1=x, 2=y, 3=z)
    Index lcid;         // Load curve ID
    Real sf;            // Scale factor

    LSDynaBodyLoad() : dof(3), lcid(0), sf(-9.81) {}
};

/**
 * @brief LS-DYNA prescribed motion data
 */
struct LSDynaPrescribedMotion {
    Index nsid;         // Node set
    int dof;            // DOF (1-6)
    int vad;            // 0=disp, 1=vel, 2=accel
    Index lcid;         // Load curve
    Real sf;            // Scale factor

    LSDynaPrescribedMotion() : nsid(0), dof(0), vad(0), lcid(0), sf(1.0) {}
};

/**
 * @brief LS-DYNA keyword format reader
 *
 * Usage:
 * ```cpp
 * LSDynaReader reader;
 * reader.read("model.k");
 *
 * auto mesh = reader.create_mesh();
 * auto materials = reader.materials();
 * auto bcs = reader.boundary_conditions();
 * ```
 */
class LSDynaReader {
public:
    LSDynaReader() = default;

    /**
     * @brief Read LS-DYNA keyword file
     * @param filename Path to .k, .dyn, or .key file
     * @return true on success
     */
    bool read(const std::string& filename);

    /**
     * @brief Create NexusSim mesh from parsed data
     * @return Shared pointer to mesh
     */
    std::shared_ptr<Mesh> create_mesh() const;

    // Accessors for parsed data
    const std::vector<LSDynaNode>& nodes() const { return nodes_; }
    const std::vector<LSDynaElement>& elements() const { return elements_; }
    const std::map<Index, LSDynaMaterial>& materials() const { return materials_; }
    const std::map<Index, LSDynaPart>& parts() const { return parts_; }
    const std::map<Index, LSDynaNodeSet>& node_sets() const { return node_sets_; }
    const std::vector<LSDynaSPC>& spcs() const { return spcs_; }
    const std::vector<LSDynaLoad>& loads() const { return loads_; }
    const std::map<Index, LSDynaLoadCurve>& load_curves() const { return load_curves_; }
    const std::map<Index, LSDynaSection>& sections() const { return sections_; }
    const std::vector<LSDynaContact>& contacts() const { return contacts_; }
    const std::map<Index, LSDynaEOS>& eos_map() const { return eos_; }
    const std::map<Index, LSDynaPartSet>& part_sets() const { return part_sets_; }
    const std::map<Index, LSDynaElementSet>& element_sets() const { return element_sets_; }
    const std::vector<LSDynaInitVelocity>& initial_velocities() const { return initial_velocities_; }
    const LSDynaControl& control() const { return control_; }
    const std::vector<LSDynaBodyLoad>& body_loads() const { return body_loads_; }
    const std::vector<LSDynaPrescribedMotion>& prescribed_motions() const { return prescribed_motions_; }
    Real d3plot_dt() const { return d3plot_dt_; }

    /**
     * @brief Get node set by ID
     */
    std::vector<Index> get_node_set(Index sid) const;

    /**
     * @brief Get elements by part ID
     */
    std::vector<Index> get_elements_by_part(Index pid) const;

    /**
     * @brief Get material properties for a part
     */
    physics::MaterialProperties get_material(Index pid) const;

    /**
     * @brief Get section properties for a part
     */
    physics::SectionProperties get_section(Index pid) const;

    /**
     * @brief Get EOS properties by EOS ID
     */
    physics::EOSProperties get_eos(Index eosid) const;

    /**
     * @brief Get model title
     */
    const std::string& title() const { return title_; }

    /**
     * @brief Print summary of parsed model
     */
    void print_summary(std::ostream& os = std::cout) const;

private:
    // Keyword handler type
    using KeywordHandler = std::function<void(std::istream&)>;

    // Initialize keyword handlers
    void init_keyword_handlers();

    // Process keyword with data lines
    void process_keyword(const std::string& keyword, const std::vector<std::string>& data_lines);

    // Parsing functions
    void parse_node(std::istream& is);
    void parse_element_solid(std::istream& is);
    void parse_element_shell(std::istream& is);
    void parse_element_beam(std::istream& is);
    void parse_part(std::istream& is);
    void parse_mat_elastic(std::istream& is);
    void parse_mat_plastic(std::istream& is);
    void parse_mat_johnson_cook(std::istream& is);
    void parse_set_node_list(std::istream& is);
    void parse_boundary_spc_set(std::istream& is);
    void parse_load_node_set(std::istream& is);
    void parse_define_curve(std::istream& is);
    void parse_title(std::istream& is);

    // Helper functions
    std::string trim(const std::string& s) const;
    std::string to_upper(const std::string& s) const;
    std::vector<std::string> split_fields(const std::string& line, int field_width = 10) const;
    bool is_keyword(const std::string& line) const;
    std::string extract_keyword(const std::string& line) const;
    bool is_comment(const std::string& line) const;
    Real parse_real(const std::string& s, Real default_val = 0.0) const;
    Index parse_index(const std::string& s, Index default_val = 0) const;
    Int parse_int(const std::string& s, Int default_val = 0) const;
    ElementType determine_solid_type(int num_nodes) const;
    ElementType determine_shell_type(int num_nodes) const;

    // Helper: parse a contact keyword
    void parse_contact(const std::string& contact_type, const std::vector<std::string>& data_lines);

    // Helper: parse a set list keyword (parts, shells, solids)
    void parse_id_set(const std::vector<std::string>& data_lines, std::vector<Index>& out_ids, Index& out_sid);

    // Parsed data
    std::string title_;

    std::vector<LSDynaNode> nodes_;
    std::vector<LSDynaElement> elements_;
    std::map<Index, LSDynaMaterial> materials_;
    std::map<Index, LSDynaPart> parts_;
    std::map<Index, LSDynaNodeSet> node_sets_;
    std::vector<LSDynaSPC> spcs_;
    std::vector<LSDynaLoad> loads_;
    std::map<Index, LSDynaLoadCurve> load_curves_;
    std::map<Index, LSDynaSection> sections_;
    std::vector<LSDynaContact> contacts_;
    std::map<Index, LSDynaEOS> eos_;
    std::map<Index, LSDynaPartSet> part_sets_;
    std::map<Index, LSDynaElementSet> element_sets_;
    std::vector<LSDynaInitVelocity> initial_velocities_;
    LSDynaControl control_;
    std::vector<LSDynaBodyLoad> body_loads_;
    std::vector<LSDynaPrescribedMotion> prescribed_motions_;
    Real d3plot_dt_ = 0.0;

    // Node ID to index mapping
    std::map<Index, Index> node_id_to_index_;

    // Contact ID counter
    Index next_contact_id_ = 1;

    // Keyword dispatch table
    std::unordered_map<std::string, KeywordHandler> keyword_handlers_;
};

// ============================================================================
// Implementation
// ============================================================================

inline void LSDynaReader::init_keyword_handlers() {
    keyword_handlers_["*NODE"] = [this](std::istream& is) { parse_node(is); };
    keyword_handlers_["*ELEMENT_SOLID"] = [this](std::istream& is) { parse_element_solid(is); };
    keyword_handlers_["*ELEMENT_SHELL"] = [this](std::istream& is) { parse_element_shell(is); };
    keyword_handlers_["*ELEMENT_BEAM"] = [this](std::istream& is) { parse_element_beam(is); };
    keyword_handlers_["*PART"] = [this](std::istream& is) { parse_part(is); };
    keyword_handlers_["*MAT_ELASTIC"] = [this](std::istream& is) { parse_mat_elastic(is); };
    keyword_handlers_["*MAT_PIECEWISE_LINEAR_PLASTICITY"] = [this](std::istream& is) { parse_mat_plastic(is); };
    keyword_handlers_["*MAT_JOHNSON_COOK"] = [this](std::istream& is) { parse_mat_johnson_cook(is); };
    keyword_handlers_["*SET_NODE_LIST"] = [this](std::istream& is) { parse_set_node_list(is); };
    keyword_handlers_["*BOUNDARY_SPC_SET"] = [this](std::istream& is) { parse_boundary_spc_set(is); };
    keyword_handlers_["*LOAD_NODE_SET"] = [this](std::istream& is) { parse_load_node_set(is); };
    keyword_handlers_["*DEFINE_CURVE"] = [this](std::istream& is) { parse_define_curve(is); };
    keyword_handlers_["*TITLE"] = [this](std::istream& is) { parse_title(is); };
}

inline bool LSDynaReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        NXS_LOG_ERROR("Failed to open file: {}", filename);
        return false;
    }

    NXS_LOG_INFO("Reading LS-DYNA keyword file: {}", filename);

    // Initialize keyword handlers
    init_keyword_handlers();

    // Clear previous data
    nodes_.clear();
    elements_.clear();
    materials_.clear();
    parts_.clear();
    node_sets_.clear();
    spcs_.clear();
    loads_.clear();
    load_curves_.clear();
    node_id_to_index_.clear();
    sections_.clear();
    contacts_.clear();
    eos_.clear();
    part_sets_.clear();
    element_sets_.clear();
    initial_velocities_.clear();
    control_ = LSDynaControl();
    body_loads_.clear();
    prescribed_motions_.clear();
    d3plot_dt_ = 0.0;
    next_contact_id_ = 1;

    // Read all lines into a vector for easier handling
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Process lines
    size_t i = 0;
    while (i < lines.size()) {
        std::string trimmed = trim(lines[i]);

        // Skip empty lines and comments
        if (trimmed.empty() || is_comment(trimmed)) {
            ++i;
            continue;
        }

        // Check for keyword
        if (is_keyword(trimmed)) {
            std::string keyword = extract_keyword(trimmed);
            std::string upper_keyword = to_upper(keyword);

            if (upper_keyword == "*END") {
                break;
            }

            ++i;  // Move past keyword line

            // Collect data lines until next keyword
            std::vector<std::string> data_lines;
            while (i < lines.size()) {
                std::string data_trimmed = trim(lines[i]);
                if (data_trimmed.empty() || is_comment(data_trimmed)) {
                    ++i;
                    continue;
                }
                if (is_keyword(data_trimmed)) {
                    break;  // Don't increment - let outer loop handle
                }
                data_lines.push_back(lines[i]);
                ++i;
            }

            // If keyword had _TITLE suffix (now stripped), first data line is the title
            {
                std::string raw_kw = to_upper(trimmed);
                auto sp = raw_kw.find(' ');
                if (sp != std::string::npos) raw_kw = raw_kw.substr(0, sp);
                if (raw_kw.length() > 6 && raw_kw.substr(raw_kw.length() - 6) == "_TITLE") {
                    if (!data_lines.empty()) {
                        data_lines.erase(data_lines.begin());
                    }
                }
            }

            // Process the keyword with collected data
            process_keyword(upper_keyword, data_lines);
        } else {
            ++i;
        }
    }

    // Build node ID to index mapping
    for (size_t j = 0; j < nodes_.size(); ++j) {
        node_id_to_index_[nodes_[j].nid] = j;
    }

    NXS_LOG_INFO("Parsed {} nodes, {} elements, {} materials, {} parts",
                 nodes_.size(), elements_.size(), materials_.size(), parts_.size());

    return true;
}

inline void LSDynaReader::parse_contact(const std::string& contact_type,
                                         const std::vector<std::string>& data_lines) {
    LSDynaContact contact;
    contact.id = next_contact_id_++;
    contact.type = contact_type;

    // Card 1: SSID, MSID, SSTYP, MSTYP, SBOXID, MBOXID, SPR, MPR
    if (!data_lines.empty()) {
        auto fields = split_fields(data_lines[0], 10);
        if (fields.size() >= 1) contact.ssid = parse_index(fields[0]);
        if (fields.size() >= 2) contact.msid = parse_index(fields[1]);
        if (fields.size() >= 3) contact.sstyp = parse_int(fields[2]);
        if (fields.size() >= 4) contact.mstyp = parse_int(fields[3]);
    }

    // Card 2: FS, FD, DC, VC, VDC, PENCHK, BT, DT
    if (data_lines.size() >= 2) {
        auto fields = split_fields(data_lines[1], 10);
        if (fields.size() >= 1) contact.fs = parse_real(fields[0]);
        if (fields.size() >= 2) contact.fd = parse_real(fields[1]);
        if (fields.size() >= 3) contact.dc = parse_real(fields[2]);
        if (fields.size() >= 4) contact.vc = parse_real(fields[3]);
    }

    // Card 3 (optional): SFS, SFM, ...
    if (data_lines.size() >= 3) {
        auto fields = split_fields(data_lines[2], 10);
        if (fields.size() >= 1) contact.sfs = parse_real(fields[0], 1.0);
        if (fields.size() >= 2) contact.sfm = parse_real(fields[1], 1.0);
        if (fields.size() >= 5) contact.soft = parse_real(fields[4]);
    }

    contacts_.push_back(contact);
}

inline void LSDynaReader::parse_id_set(const std::vector<std::string>& data_lines,
                                         std::vector<Index>& out_ids, Index& out_sid) {
    out_sid = 0;
    out_ids.clear();

    if (!data_lines.empty()) {
        auto fields = split_fields(data_lines[0], 10);
        if (fields.size() >= 1) {
            out_sid = parse_index(fields[0]);
        }
        for (size_t li = 1; li < data_lines.size(); ++li) {
            auto id_fields = split_fields(data_lines[li], 10);
            for (const auto& f : id_fields) {
                Index id = parse_index(f);
                if (id > 0) {
                    out_ids.push_back(id);
                }
            }
        }
    }
}

inline void LSDynaReader::process_keyword(const std::string& keyword,
                                          const std::vector<std::string>& data_lines) {
    if (keyword == "*NODE") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 4) continue;

            LSDynaNode node;
            node.nid = parse_index(fields[0]);
            node.x = parse_real(fields[1]);
            node.y = parse_real(fields[2]);
            node.z = parse_real(fields[3]);
            node.tc = fields.size() > 4 ? parse_int(fields[4]) : 0;
            node.rc = fields.size() > 5 ? parse_int(fields[5]) : 0;
            nodes_.push_back(node);
        }
    }
    else if (keyword == "*ELEMENT_SOLID") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 3) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);

            for (size_t i = 2; i < fields.size(); ++i) {
                Index nid = parse_index(fields[i]);
                if (nid > 0) {
                    elem.nodes.push_back(nid);
                }
            }
            elem.type = determine_solid_type(static_cast<int>(elem.nodes.size()));
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*ELEMENT_SHELL") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 4) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);

            for (size_t i = 2; i < fields.size() && i < 6; ++i) {
                Index nid = parse_index(fields[i]);
                if (nid > 0) {
                    elem.nodes.push_back(nid);
                }
            }
            elem.type = determine_shell_type(static_cast<int>(elem.nodes.size()));
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*ELEMENT_BEAM") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 5) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);
            elem.nodes.push_back(parse_index(fields[2]));
            elem.nodes.push_back(parse_index(fields[3]));
            elem.type = ElementType::Beam2;
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*PART") {
        // First line is title, second line is data
        std::string title;
        if (data_lines.size() >= 1) {
            title = trim(data_lines[0]);
        }
        if (data_lines.size() >= 2) {
            auto fields = split_fields(data_lines[1], 10);
            if (fields.size() >= 3) {
                LSDynaPart part;
                part.pid = parse_index(fields[0]);
                part.secid = parse_index(fields[1]);
                part.mid = parse_index(fields[2]);
                part.title = title;
                parts_[part.pid] = part;
            }
        }
    }
    // ========================================================================
    // Material keywords
    // ========================================================================
    else if (keyword == "*MAT_ELASTIC") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "ELASTIC";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_PIECEWISE_LINEAR_PLASTICITY") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "PIECEWISE_LINEAR_PLASTICITY";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.sigy = parse_real(fields[4]);
                mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.fail = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_JOHNSON_COOK") {
        if (data_lines.size() >= 2) {
            auto fields1 = split_fields(data_lines[0], 10);
            auto fields2 = split_fields(data_lines[1], 10);

            if (fields1.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields1[0]);
                mat.type = "JOHNSON_COOK";
                mat.ro = parse_real(fields1[1]);
                Real g = parse_real(fields1[2]);
                mat.e = parse_real(fields1[3]);
                mat.pr = parse_real(fields1[4]);
                if (mat.e == 0.0 && g > 0.0 && mat.pr > 0.0) {
                    mat.e = 2.0 * g * (1.0 + mat.pr);
                }

                if (fields2.size() >= 5) {
                    mat.a_jc = parse_real(fields2[0]);
                    mat.b_jc = parse_real(fields2[1]);
                    mat.n_jc = parse_real(fields2[2]);
                    mat.c_jc = parse_real(fields2[3]);
                    mat.m_jc = parse_real(fields2[4]);
                    mat.tm = fields2.size() > 5 ? parse_real(fields2[5]) : 0.0;
                    mat.tr = fields2.size() > 6 ? parse_real(fields2[6]) : 298.0;
                    mat.epso = fields2.size() > 7 ? parse_real(fields2[7]) : 1.0;
                }
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_RIGID") {
        // Card 1: MID, RO, E, PR, N, COUPLE, M, ALIAS
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "RIGID";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                // Card 2 (optional): CMO, CON1, CON2
                if (data_lines.size() >= 2) {
                    auto f2 = split_fields(data_lines[1], 10);
                    if (f2.size() >= 1) mat.cmo = parse_real(f2[0]);
                    if (f2.size() >= 2) mat.con1 = parse_real(f2[1]);
                    if (f2.size() >= 3) mat.con2 = parse_real(f2[2]);
                }
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_NULL") {
        // Card 1: MID, RO, PC, MU, TEROD, CEROD
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 2) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "NULL";
                mat.ro = parse_real(fields[1]);
                if (fields.size() >= 4) mat.mu = parse_real(fields[3]);
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_PLASTIC_KINEMATIC") {
        // Card 1: MID, RO, E, PR, SIGY, ETAN, BETA, SRC, SRP, FS
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "PLASTIC_KINEMATIC";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.sigy = parse_real(fields[4]);
                mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                // fields[6] = BETA (not stored)
                mat.cs_d = fields.size() > 7 ? parse_real(fields[7]) : 0.0; // SRC → CS_D
                mat.cs_q = fields.size() > 8 ? parse_real(fields[8]) : 0.0; // SRP → CS_q
                mat.fail = fields.size() > 9 ? parse_real(fields[9]) : 0.0; // FS
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_MOONEY-RIVLIN_RUBBER" || keyword == "*MAT_MOONEY_RIVLIN_RUBBER") {
        // Card 1: MID, RO, PR, A (C10), B (C01)
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "MOONEY_RIVLIN";
                mat.ro = parse_real(fields[1]);
                mat.pr = parse_real(fields[2]);
                mat.c10_mr = parse_real(fields[3]);
                mat.c01_mr = parse_real(fields[4]);
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_OGDEN_RUBBER") {
        // Card 1: MID, RO, PR, N (unused), NV, GT, REF
        // Card 2: MU1, MU2, MU3, ALPHA1, ALPHA2, ALPHA3
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 3) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "OGDEN";
                mat.ro = parse_real(fields[1]);
                mat.pr = parse_real(fields[2]);
                if (data_lines.size() >= 2) {
                    auto f2 = split_fields(data_lines[1], 10);
                    if (f2.size() >= 1) mat.mu1 = parse_real(f2[0]);
                    if (f2.size() >= 2) mat.mu2 = parse_real(f2[1]);
                    if (f2.size() >= 3) mat.mu3 = parse_real(f2[2]);
                    if (f2.size() >= 4) mat.alpha1 = parse_real(f2[3]);
                    if (f2.size() >= 5) mat.alpha2 = parse_real(f2[4]);
                    if (f2.size() >= 6) mat.alpha3 = parse_real(f2[5]);
                }
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_VISCOELASTIC") {
        // Card 1: MID, RO, BULK, G0, GI, BETA
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 6) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "VISCOELASTIC";
                mat.ro = parse_real(fields[1]);
                mat.bulk = parse_real(fields[2]);
                mat.g0 = parse_real(fields[3]);
                mat.gi = parse_real(fields[4]);
                mat.beta_ve = parse_real(fields[5]);
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_ORTHOTROPIC_ELASTIC") {
        // Card 1: MID, RO, EA, EB, EC, PRBA, PRCA, PRCB
        // Card 2: GAB, GBC, GCA
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 6) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "ORTHOTROPIC";
                mat.ro = parse_real(fields[1]);
                mat.ea = parse_real(fields[2]);
                mat.eb = parse_real(fields[3]);
                mat.ec = fields.size() > 4 ? parse_real(fields[4]) : 0.0;
                mat.prba = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.prca = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
                mat.prcb = fields.size() > 7 ? parse_real(fields[7]) : 0.0;
                if (data_lines.size() >= 2) {
                    auto f2 = split_fields(data_lines[1], 10);
                    if (f2.size() >= 1) mat.gab = parse_real(f2[0]);
                    if (f2.size() >= 2) mat.gbc = parse_real(f2[1]);
                    if (f2.size() >= 3) mat.gca = parse_real(f2[2]);
                }
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_CRUSHABLE_FOAM") {
        // Card 1: MID, RO, E, PR, LCID, TSC, DAMP
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "CRUSHABLE_FOAM";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.lcss = fields.size() > 4 ? parse_index(fields[4]) : 0;
                mat.tension_cutoff = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.damp_foam = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_HONEYCOMB") {
        // Card 1: MID, RO, E, PR, SIGY, VF, MU, BULK
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "HONEYCOMB";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.sigy = parse_real(fields[4]);
                mat.vf = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.mu = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
                mat.bulk_hc = fields.size() > 7 ? parse_real(fields[7]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_COWPER_SYMONDS") {
        // Parse as plastic kinematic with rate parameters
        // Card 1: MID, RO, E, PR, SIGY, ETAN, BETA, SRC, SRP, FS
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "COWPER_SYMONDS";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.sigy = parse_real(fields[4]);
                mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.cs_d = fields.size() > 7 ? parse_real(fields[7]) : 0.0;
                mat.cs_q = fields.size() > 8 ? parse_real(fields[8]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_FOAM") {
        // Card 1: MID, RO, E, LCID, TC, HU, BETA, DAMP
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 3) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "FOAM";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.lcss = fields.size() > 3 ? parse_index(fields[3]) : 0;
                mat.tension_cutoff = fields.size() > 4 ? parse_real(fields[4]) : 0.0;
                mat.damp_foam = fields.size() > 7 ? parse_real(fields[7]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_ELASTIC_PLASTIC_HYDRO") {
        // Card 1: MID, RO, G, SIGY, EH, PC, FS, CHATEFP
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "ELASTIC_PLASTIC_HYDRO";
                mat.ro = parse_real(fields[1]);
                mat.g_shear = parse_real(fields[2]);
                mat.sigy = parse_real(fields[3]);
                mat.etan = fields.size() > 4 ? parse_real(fields[4]) : 0.0;
                mat.bulk = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    // ========================================================================
    // Section keywords
    // ========================================================================
    else if (keyword == "*SECTION_SHELL") {
        LSDynaSection sec;
        sec.type = "SHELL";
        // Card 1: SECID, ELFORM, SHRF, NIP, PROPT, QR, ICOMP, SETYP
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) sec.secid = parse_index(fields[0]);
            if (fields.size() >= 2) sec.elform = parse_int(fields[1], 2);
            if (fields.size() >= 3) sec.shrf = parse_real(fields[2], 1.0);
            if (fields.size() >= 4) sec.nip = parse_int(fields[3], 2);
        }
        // Card 2: T1, T2, T3, T4, NLOC
        if (data_lines.size() >= 2) {
            auto fields = split_fields(data_lines[1], 10);
            if (fields.size() >= 1) sec.t1 = parse_real(fields[0]);
            if (fields.size() >= 2) sec.t2 = parse_real(fields[1]);
            if (fields.size() >= 3) sec.t3 = parse_real(fields[2]);
            if (fields.size() >= 4) sec.t4 = parse_real(fields[3]);
        }
        if (sec.secid > 0) sections_[sec.secid] = sec;
    }
    else if (keyword == "*SECTION_SOLID") {
        LSDynaSection sec;
        sec.type = "SOLID";
        // Card 1: SECID, ELFORM
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) sec.secid = parse_index(fields[0]);
            if (fields.size() >= 2) sec.elform = parse_int(fields[1], 1);
        }
        if (sec.secid > 0) sections_[sec.secid] = sec;
    }
    else if (keyword == "*SECTION_BEAM") {
        LSDynaSection sec;
        sec.type = "BEAM";
        // Card 1: SECID, ELFORM, SHRF, QR, CST, SCOOR
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) sec.secid = parse_index(fields[0]);
            if (fields.size() >= 2) sec.elform = parse_int(fields[1], 1);
            if (fields.size() >= 3) sec.shrf = parse_real(fields[2], 1.0);
            // fields[3] = QR
            if (fields.size() >= 5) sec.cst = parse_int(fields[4]);
        }
        // Card 2: TS1, TS2, TT1, TT2
        if (data_lines.size() >= 2) {
            auto fields = split_fields(data_lines[1], 10);
            if (fields.size() >= 1) sec.ts1 = parse_real(fields[0]);
            if (fields.size() >= 2) sec.ts2 = parse_real(fields[1]);
        }
        if (sec.secid > 0) sections_[sec.secid] = sec;
    }
    // ========================================================================
    // Set keywords
    // ========================================================================
    else if (keyword == "*SET_NODE_LIST") {
        LSDynaNodeSet node_set;
        node_set.sid = 0;

        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) {
                node_set.sid = parse_index(fields[0]);
            }

            // Rest are node IDs
            for (size_t li = 1; li < data_lines.size(); ++li) {
                auto node_fields = split_fields(data_lines[li], 10);
                for (const auto& f : node_fields) {
                    Index nid = parse_index(f);
                    if (nid > 0) {
                        node_set.nodes.push_back(nid);
                    }
                }
            }

            if (node_set.sid > 0) {
                node_sets_[node_set.sid] = node_set;
            }
        }
    }
    else if (keyword == "*SET_PART_LIST") {
        Index sid;
        std::vector<Index> ids;
        parse_id_set(data_lines, ids, sid);
        if (sid > 0) {
            LSDynaPartSet ps;
            ps.sid = sid;
            ps.parts = std::move(ids);
            part_sets_[ps.sid] = ps;
        }
    }
    else if (keyword == "*SET_SHELL_LIST" || keyword == "*SET_SOLID_LIST") {
        Index sid;
        std::vector<Index> ids;
        parse_id_set(data_lines, ids, sid);
        if (sid > 0) {
            LSDynaElementSet es;
            es.sid = sid;
            es.elements = std::move(ids);
            element_sets_[es.sid] = es;
        }
    }
    // ========================================================================
    // Boundary conditions
    // ========================================================================
    else if (keyword == "*BOUNDARY_SPC_SET") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaSPC spc;
                spc.nsid = parse_index(fields[0]);
                spc.cid = parse_int(fields[1]);
                spc.dofx = parse_int(fields[2]);
                spc.dofy = parse_int(fields[3]);
                spc.dofz = parse_int(fields[4]);
                spc.dofrx = fields.size() > 5 ? parse_int(fields[5]) : 0;
                spc.dofry = fields.size() > 6 ? parse_int(fields[6]) : 0;
                spc.dofrz = fields.size() > 7 ? parse_int(fields[7]) : 0;
                spcs_.push_back(spc);
            }
        }
    }
    else if (keyword == "*BOUNDARY_PRESCRIBED_MOTION_SET") {
        // Card 1: NSID, DOF, VAD, LCID, SF, VID, DEATH, BIRTH
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaPrescribedMotion pm;
                pm.nsid = parse_index(fields[0]);
                pm.dof = parse_int(fields[1]);
                pm.vad = parse_int(fields[2]);
                pm.lcid = parse_index(fields[3]);
                pm.sf = fields.size() > 4 ? parse_real(fields[4], 1.0) : 1.0;
                prescribed_motions_.push_back(pm);
            }
        }
    }
    // ========================================================================
    // Load keywords
    // ========================================================================
    else if (keyword == "*LOAD_NODE_SET") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 3) {
                LSDynaLoad load;
                load.nsid = parse_index(fields[0]);
                load.dof = parse_int(fields[1]);
                load.lcid = parse_index(fields[2]);
                load.sf = fields.size() > 3 ? parse_real(fields[3]) : 1.0;
                loads_.push_back(load);
            }
        }
    }
    else if (keyword == "*LOAD_BODY_X") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaBodyLoad bl;
            bl.dof = 1;
            if (fields.size() >= 1) bl.lcid = parse_index(fields[0]);
            if (fields.size() >= 2) bl.sf = parse_real(fields[1]);
            body_loads_.push_back(bl);
        }
    }
    else if (keyword == "*LOAD_BODY_Y") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaBodyLoad bl;
            bl.dof = 2;
            if (fields.size() >= 1) bl.lcid = parse_index(fields[0]);
            if (fields.size() >= 2) bl.sf = parse_real(fields[1]);
            body_loads_.push_back(bl);
        }
    }
    else if (keyword == "*LOAD_BODY_Z") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaBodyLoad bl;
            bl.dof = 3;
            if (fields.size() >= 1) bl.lcid = parse_index(fields[0]);
            if (fields.size() >= 2) bl.sf = parse_real(fields[1]);
            body_loads_.push_back(bl);
        }
    }
    // ========================================================================
    // Contact keywords
    // ========================================================================
    else if (keyword == "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE") {
        parse_contact("SURFACE_TO_SURFACE", data_lines);
    }
    else if (keyword == "*CONTACT_AUTOMATIC_NODES_TO_SURFACE") {
        parse_contact("NODES_TO_SURFACE", data_lines);
    }
    else if (keyword == "*CONTACT_AUTOMATIC_SINGLE_SURFACE") {
        parse_contact("SINGLE_SURFACE", data_lines);
    }
    else if (keyword == "*CONTACT_TIED_SURFACE_TO_SURFACE") {
        parse_contact("TIED", data_lines);
    }
    // ========================================================================
    // EOS keywords
    // ========================================================================
    else if (keyword == "*EOS_LINEAR_POLYNOMIAL") {
        // Card 1: EOSID, C0, C1, C2, C3, C4, C5, C6
        // Card 2: E0, V0
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaEOS eos;
            if (fields.size() >= 1) eos.eosid = parse_index(fields[0]);
            eos.type = "LINEAR_POLYNOMIAL";
            for (int j = 0; j < 7 && j + 1 < static_cast<int>(fields.size()); ++j) {
                eos.cpoly[j] = parse_real(fields[j + 1]);
            }
            if (data_lines.size() >= 2) {
                auto f2 = split_fields(data_lines[1], 10);
                if (f2.size() >= 1) eos.E0 = parse_real(f2[0]);
                if (f2.size() >= 2) eos.V0 = parse_real(f2[1], 1.0);
            }
            if (eos.eosid > 0) eos_[eos.eosid] = eos;
        }
    }
    else if (keyword == "*EOS_GRUNEISEN") {
        // Card 1: EOSID, C, S1, S2, S3, GAMAO, A, E0
        // Card 2: V0
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaEOS eos;
            if (fields.size() >= 1) eos.eosid = parse_index(fields[0]);
            eos.type = "GRUNEISEN";
            if (fields.size() >= 2) eos.C0 = parse_real(fields[1]);
            if (fields.size() >= 3) eos.S1 = parse_real(fields[2]);
            if (fields.size() >= 4) eos.S2 = parse_real(fields[3]);
            if (fields.size() >= 5) eos.S3 = parse_real(fields[4]);
            if (fields.size() >= 6) eos.gamma0 = parse_real(fields[5]);
            if (fields.size() >= 7) eos.a_coeff = parse_real(fields[6]);
            if (fields.size() >= 8) eos.E0 = parse_real(fields[7]);
            if (data_lines.size() >= 2) {
                auto f2 = split_fields(data_lines[1], 10);
                if (f2.size() >= 1) eos.V0 = parse_real(f2[0], 1.0);
            }
            if (eos.eosid > 0) eos_[eos.eosid] = eos;
        }
    }
    else if (keyword == "*EOS_JWL") {
        // Card 1: EOSID, A, B, R1, R2, OMEG, E0, V0
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaEOS eos;
            if (fields.size() >= 1) eos.eosid = parse_index(fields[0]);
            eos.type = "JWL";
            if (fields.size() >= 2) eos.A_jwl = parse_real(fields[1]);
            if (fields.size() >= 3) eos.B_jwl = parse_real(fields[2]);
            if (fields.size() >= 4) eos.R1 = parse_real(fields[3]);
            if (fields.size() >= 5) eos.R2 = parse_real(fields[4]);
            if (fields.size() >= 6) eos.omega = parse_real(fields[5]);
            if (fields.size() >= 7) eos.E0_jwl = parse_real(fields[6]);
            if (fields.size() >= 8) eos.V0_jwl = parse_real(fields[7], 1.0);
            if (eos.eosid > 0) eos_[eos.eosid] = eos;
        }
    }
    // ========================================================================
    // Initial conditions
    // ========================================================================
    else if (keyword == "*INITIAL_VELOCITY" || keyword == "*INITIAL_VELOCITY_GENERATION") {
        // Card 1: NSID, VX, VY, VZ, VRX, VRY, VRZ
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            LSDynaInitVelocity iv;
            if (fields.size() >= 1) iv.nsid = parse_index(fields[0]);
            if (fields.size() >= 2) iv.vx = parse_real(fields[1]);
            if (fields.size() >= 3) iv.vy = parse_real(fields[2]);
            if (fields.size() >= 4) iv.vz = parse_real(fields[3]);
            if (fields.size() >= 5) iv.vrx = parse_real(fields[4]);
            if (fields.size() >= 6) iv.vry = parse_real(fields[5]);
            if (fields.size() >= 7) iv.vrz = parse_real(fields[6]);
            initial_velocities_.push_back(iv);
        }
    }
    // ========================================================================
    // Control keywords
    // ========================================================================
    else if (keyword == "*CONTROL_TERMINATION") {
        // Card 1: ENDTIM, ENDCYC, DTMIN, ENDENG, ENDMAS
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) control_.termination_time = parse_real(fields[0]);
            if (fields.size() >= 3) control_.dt_min = parse_real(fields[2]);
        }
    }
    else if (keyword == "*CONTROL_TIMESTEP") {
        // Card 1: DTINIT, TSSFAC, ISDO, TSLIMT, DT2MS, LCTM, ERODE, MS1ST
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) control_.dt_init = parse_real(fields[0]);
            if (fields.size() >= 2) control_.dt_scale = parse_real(fields[1], 0.9);
            if (fields.size() >= 5) {
                Real dt2ms = parse_real(fields[4]);
                if (dt2ms != 0.0) {
                    control_.dt_ms_flag = 1;
                    control_.dt_min = dt2ms;
                }
            }
        }
    }
    // ========================================================================
    // Output keywords
    // ========================================================================
    else if (keyword == "*DATABASE_BINARY_D3PLOT") {
        // Card 1: DT, LCDT, BEAM, NPLTC
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) d3plot_dt_ = parse_real(fields[0]);
        }
    }
    // ========================================================================
    // Curve and title
    // ========================================================================
    else if (keyword == "*DEFINE_CURVE") {
        LSDynaLoadCurve curve;
        curve.lcid = 0;

        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) {
                curve.lcid = parse_index(fields[0]);
            }

            // Rest are (abscissa, ordinate) pairs
            for (size_t li = 1; li < data_lines.size(); ++li) {
                auto curve_fields = split_fields(data_lines[li], 20);
                if (curve_fields.size() >= 2) {
                    Real a = parse_real(curve_fields[0]);
                    Real o = parse_real(curve_fields[1]);
                    curve.points.emplace_back(a, o);
                }
            }

            if (curve.lcid > 0) {
                load_curves_[curve.lcid] = curve;
            }
        }
    }
    else if (keyword == "*TITLE") {
        if (!data_lines.empty()) {
            title_ = trim(data_lines[0]);
        }
    }
    // Other keywords are silently ignored
}

inline void LSDynaReader::parse_node(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) {
            break;
        }

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 4) continue;

        LSDynaNode node;
        node.nid = parse_index(fields[0]);
        node.x = parse_real(fields[1]);
        node.y = parse_real(fields[2]);
        node.z = parse_real(fields[3]);
        node.tc = fields.size() > 4 ? parse_int(fields[4]) : 0;
        node.rc = fields.size() > 5 ? parse_int(fields[5]) : 0;

        nodes_.push_back(node);
    }
}

inline void LSDynaReader::parse_element_solid(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 3) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);

        for (size_t i = 2; i < fields.size(); ++i) {
            Index nid = parse_index(fields[i]);
            if (nid > 0) {
                elem.nodes.push_back(nid);
            }
        }

        elem.type = determine_solid_type(static_cast<int>(elem.nodes.size()));
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_element_shell(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 4) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);

        for (size_t i = 2; i < fields.size() && i < 6; ++i) {
            Index nid = parse_index(fields[i]);
            if (nid > 0) {
                elem.nodes.push_back(nid);
            }
        }

        elem.type = determine_shell_type(static_cast<int>(elem.nodes.size()));
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_element_beam(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 5) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);
        elem.nodes.push_back(parse_index(fields[2]));
        elem.nodes.push_back(parse_index(fields[3]));

        elem.type = ElementType::Beam2;
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_part(std::istream& is) {
    std::string line;
    std::string title;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (!is_keyword(trimmed) && !trimmed.empty()) {
            title = trimmed;
        }
    }

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 3) {
            LSDynaPart part;
            part.pid = parse_index(fields[0]);
            part.secid = parse_index(fields[1]);
            part.mid = parse_index(fields[2]);
            part.title = title;
            parts_[part.pid] = part;
        }
    }
}

inline void LSDynaReader::parse_mat_elastic(std::istream& is) {
    std::string line;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 4) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "ELASTIC";
            mat.ro = parse_real(fields[1]);
            mat.e = parse_real(fields[2]);
            mat.pr = parse_real(fields[3]);
            materials_[mat.mid] = mat;
        }
    }
}

inline void LSDynaReader::parse_mat_plastic(std::istream& is) {
    std::string line;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "PIECEWISE_LINEAR_PLASTICITY";
            mat.ro = parse_real(fields[1]);
            mat.e = parse_real(fields[2]);
            mat.pr = parse_real(fields[3]);
            mat.sigy = parse_real(fields[4]);
            mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
            mat.fail = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
            materials_[mat.mid] = mat;
        }
    }

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;
    }
}

inline void LSDynaReader::parse_mat_johnson_cook(std::istream& is) {
    std::string line;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "JOHNSON_COOK";
            mat.ro = parse_real(fields[1]);
            Real g = parse_real(fields[2]);
            mat.e = parse_real(fields[3]);
            mat.pr = parse_real(fields[4]);
            if (mat.e == 0.0 && g > 0.0 && mat.pr > 0.0) {
                mat.e = 2.0 * g * (1.0 + mat.pr);
            }
            materials_[mat.mid] = mat;
        }
    }

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5 && materials_.count(parse_index(split_fields(trimmed, 10)[0])) == 0) {
            if (!materials_.empty()) {
                auto& mat = materials_.rbegin()->second;
                mat.a_jc = parse_real(fields[0]);
                mat.b_jc = parse_real(fields[1]);
                mat.n_jc = parse_real(fields[2]);
                mat.c_jc = parse_real(fields[3]);
                mat.m_jc = parse_real(fields[4]);
                mat.tm = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.tr = fields.size() > 6 ? parse_real(fields[6]) : 298.0;
                mat.epso = fields.size() > 7 ? parse_real(fields[7]) : 1.0;
            }
        }
    }
}

inline void LSDynaReader::parse_set_node_list(std::istream& is) {
    std::string line;
    LSDynaNodeSet node_set;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 1) {
            node_set.sid = parse_index(fields[0]);
        }
    }

    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 10);
        for (const auto& f : fields) {
            Index nid = parse_index(f);
            if (nid > 0) {
                node_set.nodes.push_back(nid);
            }
        }
    }

    if (node_set.sid > 0) {
        node_sets_[node_set.sid] = node_set;
    }
}

inline void LSDynaReader::parse_boundary_spc_set(std::istream& is) {
    std::string line;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaSPC spc;
            spc.nsid = parse_index(fields[0]);
            spc.cid = parse_int(fields[1]);
            spc.dofx = parse_int(fields[2]);
            spc.dofy = parse_int(fields[3]);
            spc.dofz = parse_int(fields[4]);
            spc.dofrx = fields.size() > 5 ? parse_int(fields[5]) : 0;
            spc.dofry = fields.size() > 6 ? parse_int(fields[6]) : 0;
            spc.dofrz = fields.size() > 7 ? parse_int(fields[7]) : 0;
            spcs_.push_back(spc);
        }
    }
}

inline void LSDynaReader::parse_load_node_set(std::istream& is) {
    std::string line;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 3) {
            LSDynaLoad load;
            load.nsid = parse_index(fields[0]);
            load.dof = parse_int(fields[1]);
            load.lcid = parse_index(fields[2]);
            load.sf = fields.size() > 3 ? parse_real(fields[3]) : 1.0;
            loads_.push_back(load);
        }
    }
}

inline void LSDynaReader::parse_define_curve(std::istream& is) {
    std::string line;
    LSDynaLoadCurve curve;

    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 1) {
            curve.lcid = parse_index(fields[0]);
        }
    }

    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 20);
        if (fields.size() >= 2) {
            Real a = parse_real(fields[0]);
            Real o = parse_real(fields[1]);
            curve.points.emplace_back(a, o);
        }
    }

    if (curve.lcid > 0) {
        load_curves_[curve.lcid] = curve;
    }
}

inline void LSDynaReader::parse_title(std::istream& is) {
    std::string line;
    if (std::getline(is, line)) {
        title_ = trim(line);
    }
}

inline std::shared_ptr<Mesh> LSDynaReader::create_mesh() const {
    if (nodes_.empty()) {
        NXS_LOG_ERROR("No nodes parsed");
        return nullptr;
    }

    auto mesh = std::make_shared<Mesh>(nodes_.size());

    // Set node coordinates
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        mesh->set_node_coordinates(i, Vec3r{node.x, node.y, node.z});
    }

    // Group elements by type and part
    std::map<std::pair<ElementType, Index>, std::vector<const LSDynaElement*>> element_groups;
    for (const auto& elem : elements_) {
        element_groups[{elem.type, elem.pid}].push_back(&elem);
    }

    // Create element blocks
    for (const auto& [key, elems] : element_groups) {
        auto [type, pid] = key;
        size_t nodes_per_elem = elems[0]->nodes.size();

        std::string block_name = "part_" + std::to_string(pid);
        Index block_id = mesh->add_element_block(block_name, type, elems.size(), nodes_per_elem);
        auto& block = mesh->element_block(block_id);

        // Set connectivity
        for (size_t ei = 0; ei < elems.size(); ++ei) {
            auto elem_nodes = block.element_nodes(ei);
            const auto& elem = *elems[ei];

            for (size_t ni = 0; ni < elem.nodes.size(); ++ni) {
                Index nid = elem.nodes[ni];
                auto it = node_id_to_index_.find(nid);
                if (it != node_id_to_index_.end()) {
                    elem_nodes[ni] = it->second;
                } else {
                    NXS_LOG_WARN("Node ID {} not found for element {}", nid, elem.eid);
                    elem_nodes[ni] = 0;
                }
            }

            // Set material and part IDs
            block.material_ids.component(0)[ei] = pid;
            block.part_ids.component(0)[ei] = pid;
        }
    }

    // Add node sets
    for (const auto& [sid, ns] : node_sets_) {
        std::vector<Index> indices;
        for (Index nid : ns.nodes) {
            auto it = node_id_to_index_.find(nid);
            if (it != node_id_to_index_.end()) {
                indices.push_back(it->second);
            }
        }
        mesh->add_node_set("set_" + std::to_string(sid), indices);
    }

    return mesh;
}

inline std::vector<Index> LSDynaReader::get_node_set(Index sid) const {
    auto it = node_sets_.find(sid);
    if (it != node_sets_.end()) {
        std::vector<Index> indices;
        for (Index nid : it->second.nodes) {
            auto nit = node_id_to_index_.find(nid);
            if (nit != node_id_to_index_.end()) {
                indices.push_back(nit->second);
            }
        }
        return indices;
    }
    return {};
}

inline std::vector<Index> LSDynaReader::get_elements_by_part(Index pid) const {
    std::vector<Index> result;
    for (size_t i = 0; i < elements_.size(); ++i) {
        if (elements_[i].pid == pid) {
            result.push_back(i);
        }
    }
    return result;
}

inline physics::MaterialProperties LSDynaReader::get_material(Index pid) const {
    physics::MaterialProperties props;

    auto pit = parts_.find(pid);
    if (pit == parts_.end()) {
        return props;
    }

    Index mid = pit->second.mid;
    auto mit = materials_.find(mid);
    if (mit == materials_.end()) {
        return props;
    }

    const auto& mat = mit->second;
    props.density = mat.ro;
    props.E = mat.e;
    props.nu = mat.pr;

    if (mat.type == "ELASTIC") {
        // Already set E, nu
    }
    else if (mat.type == "PIECEWISE_LINEAR_PLASTICITY") {
        props.yield_stress = mat.sigy;
        props.tangent_modulus = mat.etan;
    }
    else if (mat.type == "JOHNSON_COOK") {
        props.JC_A = mat.a_jc;
        props.JC_B = mat.b_jc;
        props.JC_n = mat.n_jc;
        props.JC_C = mat.c_jc;
        props.JC_m = mat.m_jc;
        props.JC_T_melt = mat.tm;
        props.JC_T_room = mat.tr > 0 ? mat.tr : 298.0;
        props.JC_eps_dot_ref = mat.epso;
        props.yield_stress = mat.a_jc;
    }
    else if (mat.type == "RIGID") {
        // Rigid: set E, nu, density
    }
    else if (mat.type == "NULL") {
        // Null material: pressure only
    }
    else if (mat.type == "PLASTIC_KINEMATIC") {
        props.yield_stress = mat.sigy;
        props.tangent_modulus = mat.etan;
        props.CS_D = mat.cs_d;
        props.CS_q = mat.cs_q;
    }
    else if (mat.type == "MOONEY_RIVLIN") {
        props.C10 = mat.c10_mr;
        props.C01 = mat.c01_mr;
    }
    else if (mat.type == "OGDEN") {
        props.ogden_mu[0] = mat.mu1;
        props.ogden_mu[1] = mat.mu2;
        props.ogden_mu[2] = mat.mu3;
        props.ogden_alpha[0] = mat.alpha1;
        props.ogden_alpha[1] = mat.alpha2;
        props.ogden_alpha[2] = mat.alpha3;
        int n = 0;
        if (mat.mu1 != 0.0) n = 1;
        if (mat.mu2 != 0.0) n = 2;
        if (mat.mu3 != 0.0) n = 3;
        props.ogden_nterms = n;
    }
    else if (mat.type == "VISCOELASTIC") {
        props.K = mat.bulk;
        props.G = mat.g0;
        props.prony_g[0] = mat.gi;
        props.prony_tau[0] = (mat.beta_ve > 0.0) ? 1.0 / mat.beta_ve : 1.0;
        props.prony_nterms = 1;
    }
    else if (mat.type == "ORTHOTROPIC") {
        props.E1 = mat.ea;
        props.E2 = mat.eb;
        props.E3 = mat.ec;
        props.nu12 = mat.prba;
        props.nu13 = mat.prca;
        props.nu23 = mat.prcb;
        props.G12 = mat.gab;
        props.G23 = mat.gbc;
        props.G13 = mat.gca;
    }
    else if (mat.type == "CRUSHABLE_FOAM") {
        props.foam_E_crush = mat.e_crush;
        props.foam_densification = mat.densification;
    }
    else if (mat.type == "HONEYCOMB") {
        props.yield_stress = mat.sigy;
    }
    else if (mat.type == "COWPER_SYMONDS") {
        props.yield_stress = mat.sigy;
        props.tangent_modulus = mat.etan;
        props.CS_D = mat.cs_d;
        props.CS_q = mat.cs_q;
    }
    else if (mat.type == "FOAM") {
        // Foam: E set from card
    }
    else if (mat.type == "ELASTIC_PLASTIC_HYDRO") {
        props.G = mat.g_shear;
        props.yield_stress = mat.sigy;
        props.K = mat.bulk;
    }

    // Compute derived properties if E and nu are set
    if (props.E > 0.0 && props.nu >= 0.0 && props.nu < 0.5) {
        props.compute_derived();
    }

    return props;
}

inline physics::SectionProperties LSDynaReader::get_section(Index pid) const {
    physics::SectionProperties sec_props;

    auto pit = parts_.find(pid);
    if (pit == parts_.end()) return sec_props;

    Index secid = pit->second.secid;
    auto sit = sections_.find(secid);
    if (sit == sections_.end()) return sec_props;

    const auto& sec = sit->second;
    sec_props.id = static_cast<int>(sec.secid);

    if (sec.type == "SHELL") {
        sec_props.type = physics::SectionType::ShellUniform;
        sec_props.thickness = sec.t1;
        sec_props.thickness_nodes[0] = sec.t1;
        sec_props.thickness_nodes[1] = sec.t2 > 0 ? sec.t2 : sec.t1;
        sec_props.thickness_nodes[2] = sec.t3 > 0 ? sec.t3 : sec.t1;
        sec_props.thickness_nodes[3] = sec.t4 > 0 ? sec.t4 : sec.t1;
        sec_props.num_ip_thickness = sec.nip;
        // If thicknesses differ, use variable type
        if ((sec.t2 > 0 && sec.t2 != sec.t1) ||
            (sec.t3 > 0 && sec.t3 != sec.t1) ||
            (sec.t4 > 0 && sec.t4 != sec.t1)) {
            sec_props.type = physics::SectionType::ShellVariable;
        }
    }
    else if (sec.type == "BEAM") {
        if (sec.cst == 1) {
            sec_props.type = physics::SectionType::BeamRectangular;
            sec_props.width = sec.ts1;
            sec_props.height = sec.ts2;
        } else {
            sec_props.type = physics::SectionType::BeamCircular;
            sec_props.diameter = sec.ts1;
        }
    }
    else if (sec.type == "SOLID") {
        // Solid sections don't map to beam/shell section types directly
        sec_props.type = physics::SectionType::ShellUniform;
    }

    sec_props.compute();
    return sec_props;
}

inline physics::EOSProperties LSDynaReader::get_eos(Index eosid) const {
    physics::EOSProperties props;

    auto it = eos_.find(eosid);
    if (it == eos_.end()) return props;

    const auto& eos = it->second;

    if (eos.type == "GRUNEISEN") {
        props.type = physics::EOSType::Gruneisen;
        props.C0 = eos.C0;
        props.S1 = eos.S1;
        props.S2 = eos.S2;
        props.S3 = eos.S3;
        props.gamma0 = eos.gamma0;
        props.a_coeff = eos.a_coeff;
    }
    else if (eos.type == "JWL") {
        props.type = physics::EOSType::JWL;
        props.A_jwl = eos.A_jwl;
        props.B_jwl = eos.B_jwl;
        props.R1 = eos.R1;
        props.R2 = eos.R2;
        props.omega = eos.omega;
    }
    else if (eos.type == "LINEAR_POLYNOMIAL") {
        props.type = physics::EOSType::LinearPolynomial;
        for (int i = 0; i < 7; ++i) {
            props.C_poly[i] = eos.cpoly[i];
        }
    }

    return props;
}

inline void LSDynaReader::print_summary(std::ostream& os) const {
    os << "LS-DYNA Model Summary\n";
    os << "=====================\n";
    if (!title_.empty()) {
        os << "Title: " << title_ << "\n";
    }
    os << "Nodes: " << nodes_.size() << "\n";
    os << "Elements: " << elements_.size() << "\n";
    os << "Materials: " << materials_.size() << "\n";
    os << "Parts: " << parts_.size() << "\n";
    os << "Sections: " << sections_.size() << "\n";
    os << "Node sets: " << node_sets_.size() << "\n";
    os << "Part sets: " << part_sets_.size() << "\n";
    os << "Element sets: " << element_sets_.size() << "\n";
    os << "SPCs: " << spcs_.size() << "\n";
    os << "Loads: " << loads_.size() << "\n";
    os << "Body loads: " << body_loads_.size() << "\n";
    os << "Prescribed motions: " << prescribed_motions_.size() << "\n";
    os << "Load curves: " << load_curves_.size() << "\n";
    os << "Contacts: " << contacts_.size() << "\n";
    os << "EOS: " << eos_.size() << "\n";
    os << "Initial velocities: " << initial_velocities_.size() << "\n";
    os << "Termination time: " << control_.termination_time << "\n";
    if (d3plot_dt_ > 0.0) {
        os << "D3PLOT interval: " << d3plot_dt_ << "\n";
    }

    // Element type breakdown
    std::map<ElementType, size_t> type_counts;
    for (const auto& elem : elements_) {
        type_counts[elem.type]++;
    }
    os << "\nElement Types:\n";
    for (const auto& [type, count] : type_counts) {
        os << "  " << to_string(type) << ": " << count << "\n";
    }
}

// Helper function implementations
inline std::string LSDynaReader::trim(const std::string& s) const {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::string LSDynaReader::to_upper(const std::string& s) const {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

inline std::vector<std::string> LSDynaReader::split_fields(const std::string& line, int field_width) const {
    std::vector<std::string> fields;

    // LS-DYNA uses fixed-width fields (typically 8 or 10 characters)
    // But we also need to handle comma-separated format
    if (line.find(',') != std::string::npos) {
        // Comma-separated
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {
            fields.push_back(trim(field));
        }
    } else {
        // Fixed-width or space-separated
        // Try space-separated first
        std::istringstream iss(line);
        std::string field;
        while (iss >> field) {
            fields.push_back(field);
        }
    }

    return fields;
}

inline bool LSDynaReader::is_keyword(const std::string& line) const {
    return !line.empty() && line[0] == '*';
}

inline std::string LSDynaReader::extract_keyword(const std::string& line) const {
    std::string kw = line;

    // Remove any trailing content after space
    auto space = kw.find(' ');
    if (space != std::string::npos) {
        kw = kw.substr(0, space);
    }

    // Check for _TITLE suffix and remove it
    if (kw.length() > 6 && kw.substr(kw.length() - 6) == "_TITLE") {
        kw = kw.substr(0, kw.length() - 6);
    }

    return kw;
}

inline bool LSDynaReader::is_comment(const std::string& line) const {
    return !line.empty() && line[0] == '$';
}

inline Real LSDynaReader::parse_real(const std::string& s, Real default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return std::stod(trimmed);
    } catch (...) {
        return default_val;
    }
}

inline Index LSDynaReader::parse_index(const std::string& s, Index default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return static_cast<Index>(std::stoul(trimmed));
    } catch (...) {
        return default_val;
    }
}

inline Int LSDynaReader::parse_int(const std::string& s, Int default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return std::stoi(trimmed);
    } catch (...) {
        return default_val;
    }
}

inline ElementType LSDynaReader::determine_solid_type(int num_nodes) const {
    switch (num_nodes) {
        case 4:  return ElementType::Tet4;
        case 6:  return ElementType::Wedge6;
        case 8:  return ElementType::Hex8;
        case 10: return ElementType::Tet10;
        case 20: return ElementType::Hex20;
        default: return ElementType::Hex8;  // Default
    }
}

inline ElementType LSDynaReader::determine_shell_type(int num_nodes) const {
    switch (num_nodes) {
        case 3: return ElementType::Shell3;
        case 4: return ElementType::Shell4;
        default: return ElementType::Shell4;
    }
}

} // namespace io
} // namespace nxs
