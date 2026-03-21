#pragma once

/**
 * @file constraints_wave43.hpp
 * @brief Wave 43: FXBODY superelement embedding, RLINK velocity-level rigid links,
 *        and GroupSet set-algebra infrastructure
 *
 * Features:
 * - FXBODYConstraint: Embed Craig-Bampton reduced superelements into full FE models
 * - RLINKConstraint: Velocity-level rigid links (Type0/1/2/10)
 * - GroupSet + GroupSetManager: Named set algebra (union, intersect, difference, complement)
 *
 * Reference: OpenRadioss /engine/source/constraints/fxbody, rlink, sets
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <sstream>
#include <cctype>

namespace nxs {
namespace fem {

// ============================================================================
// (a) FXBODY — Flexible Body Superelement
// ============================================================================

/**
 * @brief Reduced superelement data (Craig-Bampton or Guyan condensation).
 *
 * Matrices are stored row-major as flat std::vector<Real>:
 *   K_reduced[i*N + j]  where N = num_interface_dofs
 *   recovery_matrix[i*N + j]  where rows = num_internal_modes, cols = N
 */
struct SuperelementData {
    int id = 0;
    std::string name;

    /// Number of interface (boundary) DOFs
    int num_interface_dofs = 0;

    /// Number of internal (retained) modes (for recovery only)
    int num_internal_modes = 0;

    /// Global DOF indices for each interface DOF (size = num_interface_dofs)
    std::vector<Index> interface_dof_ids;

    /// Reduced stiffness matrix, row-major, size = num_interface_dofs^2
    std::vector<Real> K_reduced;

    /// Reduced mass matrix, row-major, size = num_interface_dofs^2
    std::vector<Real> M_reduced;

    /// Recovery matrix: maps interface displacements -> internal mode amplitudes
    /// Size = num_internal_modes * num_interface_dofs, row-major
    std::vector<Real> recovery_matrix;

    SuperelementData() = default;

    SuperelementData(int id_, std::string name_,
                     int n_intf, int n_int = 0)
        : id(id_), name(std::move(name_))
        , num_interface_dofs(n_intf)
        , num_internal_modes(n_int)
    {
        interface_dof_ids.resize(n_intf, 0);
        K_reduced.assign(n_intf * n_intf, Real(0));
        M_reduced.assign(n_intf * n_intf, Real(0));
        recovery_matrix.assign(n_int * n_intf, Real(0));
    }
};

/**
 * @brief FXBODY constraint: assembles a reduced superelement into the global system.
 *
 * The superelement contributes entries to the global stiffness / mass at its
 * interface DOFs.  Internal displacements can be recovered via the recovery matrix.
 */
class FXBODYConstraint {
public:
    explicit FXBODYConstraint(SuperelementData data)
        : data_(std::move(data)) {}

    const SuperelementData& data() const { return data_; }
    int id() const { return data_.id; }
    const std::string& name() const { return data_.name; }

    /**
     * @brief Add K_reduced entries to the diagonal and off-diagonal storage.
     *
     * @param K_global_diag   Diagonal of the global stiffness (indexed by global DOF).
     *                        Size >= max(interface_dof_ids)+1.
     * @param K_global_offdiag  Symmetric off-diagonal entries stored as a flat array
     *                          of size total_dofs * total_dofs (row-major) for simplicity.
     *                          Pass nullptr to skip off-diagonal assembly.
     * @param total_dofs      Number of global DOFs (needed for row-major addressing).
     */
    void assemble_stiffness(Real* K_global_diag,
                            Real* K_global_offdiag,
                            Index total_dofs) const
    {
        const int N = data_.num_interface_dofs;
        for (int i = 0; i < N; ++i) {
            Index gi = data_.interface_dof_ids[i];
            // Diagonal
            K_global_diag[gi] += data_.K_reduced[i * N + i];
            // Off-diagonal
            if (K_global_offdiag) {
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    Index gj = data_.interface_dof_ids[j];
                    K_global_offdiag[gi * total_dofs + gj] +=
                        data_.K_reduced[i * N + j];
                }
            }
        }
    }

    /**
     * @brief Add M_reduced entries to the global diagonal mass.
     *
     * Lumped-mass assembly: only the diagonal of M_reduced is added to K_global_diag.
     * Full consistent mass uses the same off-diagonal pattern as stiffness.
     */
    void assemble_mass(Real* M_global_diag,
                       Real* M_global_offdiag,
                       Index total_dofs) const
    {
        const int N = data_.num_interface_dofs;
        for (int i = 0; i < N; ++i) {
            Index gi = data_.interface_dof_ids[i];
            M_global_diag[gi] += data_.M_reduced[i * N + i];
            if (M_global_offdiag) {
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    Index gj = data_.interface_dof_ids[j];
                    M_global_offdiag[gi * total_dofs + gj] +=
                        data_.M_reduced[i * N + j];
                }
            }
        }
    }

    /**
     * @brief Recover internal mode amplitudes from interface displacements.
     *
     * q_internal = recovery_matrix * u_interface
     *
     * @param interface_displacements  Length = num_interface_dofs
     * @return vector of length num_internal_modes
     */
    std::vector<Real> recover_internal(
        const std::vector<Real>& interface_displacements) const
    {
        const int N = data_.num_interface_dofs;
        const int M = data_.num_internal_modes;
        std::vector<Real> q(M, Real(0));

        for (int i = 0; i < M; ++i) {
            Real sum = Real(0);
            for (int j = 0; j < N; ++j) {
                sum += data_.recovery_matrix[i * N + j] *
                       interface_displacements[j];
            }
            q[i] = sum;
        }
        return q;
    }

    /**
     * @brief Compute interface forces from interface displacements.
     *
     * f_interface = K_reduced * u_interface
     *
     * @param interface_displacements  Length = num_interface_dofs
     * @return force vector of length num_interface_dofs
     */
    std::vector<Real> compute_interface_forces(
        const std::vector<Real>& interface_displacements) const
    {
        const int N = data_.num_interface_dofs;
        std::vector<Real> f(N, Real(0));

        for (int i = 0; i < N; ++i) {
            Real sum = Real(0);
            for (int j = 0; j < N; ++j) {
                sum += data_.K_reduced[i * N + j] *
                       interface_displacements[j];
            }
            f[i] = sum;
        }
        return f;
    }

private:
    SuperelementData data_;
};

// ============================================================================
// (b) RLINK — Velocity-Level Rigid Link
// ============================================================================

/**
 * @brief RLINK coupling types (OpenRadioss /TYPE numbering convention).
 *
 * Type0  : All 6 DOFs coupled (v_slave = v_master + omega x r)
 * Type1  : Translations only (v_slave_trans = v_master_trans)
 * Type2  : User-selected DOFs (controlled by released_dofs[6])
 * Type10 : Spherical — shared translation, free rotation
 */
enum class RLINKType {
    Type0  = 0,   ///< Full rigid link (translation + rotation)
    Type1  = 1,   ///< Translations only
    Type2  = 2,   ///< User-selected DOFs
    Type10 = 10   ///< Spherical (shared translation only)
};

/**
 * @brief Velocity-level rigid link constraint.
 *
 * Unlike displacement-level RBE2, RLINK enforces v_slave = v_master at each
 * time step.  This allows time integration schemes to handle the constraint
 * incrementally without drift accumulation.
 */
class RLINKConstraint {
public:
    Index master_node = 0;
    std::vector<Index> slave_nodes;
    RLINKType type = RLINKType::Type0;

    /// Per-DOF release flags [ux,uy,uz,rx,ry,rz].
    /// true = released (free), false = constrained.  Used by Type2 only.
    bool released_dofs[6] = {false, false, false, false, false, false};

    RLINKConstraint() = default;

    RLINKConstraint(Index master, std::vector<Index> slaves, RLINKType t)
        : master_node(master), slave_nodes(std::move(slaves)), type(t) {}

    /**
     * @brief Enforce velocity constraint on all slave nodes.
     *
     * Layout: velocities[6*node + dof], where dof 0-2 = translational,
     *         dof 3-5 = rotational.  positions[3*node + d] for coordinates.
     *
     * @param velocities  Packed velocity array [6*num_nodes]
     * @param positions   Packed position array [3*num_nodes] (for omega x r)
     * @param num_nodes   Total number of nodes in the model
     */
    void apply_velocity_constraint(Real* velocities,
                                   const Real* positions,
                                   Index num_nodes) const
    {
        (void)num_nodes;  // Size guard left to caller

        Index m = master_node;
        Real vm[3] = { velocities[6*m+0], velocities[6*m+1], velocities[6*m+2] };
        Real om[3] = { velocities[6*m+3], velocities[6*m+4], velocities[6*m+5] };

        for (Index s : slave_nodes) {
            switch (type) {
                case RLINKType::Type0: {
                    // v_s = v_m + omega_m x r_{ms}
                    Real r[3] = {
                        positions[3*s+0] - positions[3*m+0],
                        positions[3*s+1] - positions[3*m+1],
                        positions[3*s+2] - positions[3*m+2]
                    };
                    velocities[6*s+0] = vm[0] + om[1]*r[2] - om[2]*r[1];
                    velocities[6*s+1] = vm[1] + om[2]*r[0] - om[0]*r[2];
                    velocities[6*s+2] = vm[2] + om[0]*r[1] - om[1]*r[0];
                    // Rotational DOFs also coupled
                    velocities[6*s+3] = om[0];
                    velocities[6*s+4] = om[1];
                    velocities[6*s+5] = om[2];
                    break;
                }
                case RLINKType::Type1:
                case RLINKType::Type10: {
                    // Translation only (spherical: no rotational coupling)
                    velocities[6*s+0] = vm[0];
                    velocities[6*s+1] = vm[1];
                    velocities[6*s+2] = vm[2];
                    // Leave rotational DOFs free
                    break;
                }
                case RLINKType::Type2: {
                    // Enforce only non-released DOFs (translations)
                    for (int d = 0; d < 3; ++d) {
                        if (!released_dofs[d]) {
                            velocities[6*s+d] = vm[d];
                        }
                    }
                    // Rotational DOFs (3-5): enforce if not released and master
                    // has rotational velocity
                    for (int d = 3; d < 6; ++d) {
                        if (!released_dofs[d]) {
                            velocities[6*s+d] = om[d-3];
                        }
                    }
                    break;
                }
            }
        }
    }

    /**
     * @brief Compute reaction forces at slave nodes due to the constraint.
     *
     * Simple inertial estimate: f_reaction[s] = mass[s] * (a_master - a_slave).
     * Layout: accelerations[6*node + dof], masses[node].
     *
     * @return Flat array of size 6 * slave_nodes.size()
     */
    std::vector<Real> compute_constraint_forces(
        const Real* accelerations,
        const Real* masses) const
    {
        const std::size_t ns = slave_nodes.size();
        std::vector<Real> forces(6 * ns, Real(0));

        Index m = master_node;
        for (std::size_t si = 0; si < ns; ++si) {
            Index s = slave_nodes[si];
            Real ms = masses[s];
            for (int d = 0; d < 6; ++d) {
                forces[6*si + d] =
                    ms * (accelerations[6*m + d] - accelerations[6*s + d]);
            }
        }
        return forces;
    }
};

// ============================================================================
// (c) GroupSet — Set Algebra on Node/Element/Part/Surface Groups
// ============================================================================

/// Type of entities stored in a GroupSet
enum class SetType {
    NodeSet,
    ElementSet,
    PartSet,
    SurfaceSet
};

/**
 * @brief Sorted ID set with full set-algebra operations.
 *
 * Internally stores a sorted, deduplicated std::vector<int>.
 * All mutating operations keep the invariant that ids_ is sorted+unique.
 */
class GroupSet {
public:
    GroupSet() = default;

    explicit GroupSet(SetType type) : type_(type) {}

    GroupSet(SetType type, std::vector<int> ids)
        : type_(type), ids_(std::move(ids))
    {
        normalize();
    }

    SetType set_type() const { return type_; }
    std::size_t size() const { return ids_.size(); }
    bool empty() const { return ids_.empty(); }
    const std::vector<int>& ids() const { return ids_; }

    /// Add an ID (maintains sorted+unique invariant)
    void add(int id) {
        auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
        if (it == ids_.end() || *it != id) {
            ids_.insert(it, id);
        }
    }

    /// Remove an ID (no-op if not present)
    void remove(int id) {
        auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
        if (it != ids_.end() && *it == id) {
            ids_.erase(it);
        }
    }

    /// Check membership
    bool contains(int id) const {
        return std::binary_search(ids_.begin(), ids_.end(), id);
    }

    /// Set union: return all IDs in this OR other
    GroupSet union_with(const GroupSet& other) const {
        GroupSet result(type_);
        std::set_union(
            ids_.begin(), ids_.end(),
            other.ids_.begin(), other.ids_.end(),
            std::back_inserter(result.ids_));
        return result;
    }

    /// Set intersection: return IDs in this AND other
    GroupSet intersect(const GroupSet& other) const {
        GroupSet result(type_);
        std::set_intersection(
            ids_.begin(), ids_.end(),
            other.ids_.begin(), other.ids_.end(),
            std::back_inserter(result.ids_));
        return result;
    }

    /// Set difference: return IDs in this but NOT in other
    GroupSet difference(const GroupSet& other) const {
        GroupSet result(type_);
        std::set_difference(
            ids_.begin(), ids_.end(),
            other.ids_.begin(), other.ids_.end(),
            std::back_inserter(result.ids_));
        return result;
    }

    /// Complement: return IDs in universe but NOT in this
    GroupSet complement(const GroupSet& universe) const {
        return universe.difference(*this);
    }

private:
    /// Ensure ids_ is sorted and contains no duplicates
    void normalize() {
        std::sort(ids_.begin(), ids_.end());
        ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
    }

    SetType type_ = SetType::NodeSet;
    std::vector<int> ids_;
};

/**
 * @brief Manager for named GroupSets with expression evaluation.
 *
 * Supports expressions of the form:
 *   "SET_A + SET_B - SET_C"
 *
 * where '+' means union and '-' means difference, evaluated left-to-right.
 * All named sets must have been registered via create_set() before evaluation.
 */
class GroupSetManager {
public:
    GroupSetManager() = default;

    /**
     * @brief Create (or replace) a named set of the given type.
     * @return Reference to the newly created empty set.
     */
    GroupSet& create_set(const std::string& name, SetType type) {
        sets_[name] = GroupSet(type);
        return sets_[name];
    }

    /**
     * @brief Retrieve a named set (throws if not found).
     */
    GroupSet& get_set(const std::string& name) {
        auto it = sets_.find(name);
        if (it == sets_.end()) {
            throw std::runtime_error("GroupSetManager: set '" + name + "' not found");
        }
        return it->second;
    }

    const GroupSet& get_set(const std::string& name) const {
        auto it = sets_.find(name);
        if (it == sets_.end()) {
            throw std::runtime_error("GroupSetManager: set '" + name + "' not found");
        }
        return it->second;
    }

    bool has_set(const std::string& name) const {
        return sets_.count(name) > 0;
    }

    std::size_t num_sets() const { return sets_.size(); }

    /**
     * @brief Evaluate a simple set expression: "SET1 + SET2 - SET3 ..."
     *
     * Grammar (left-to-right, no precedence beyond order):
     *   expr  := name (op name)*
     *   op    := '+' | '-'
     *
     * '+' = union, '-' = difference.
     * Whitespace around tokens is ignored.
     *
     * @param expr  Expression string
     * @return Resulting GroupSet (type inherited from the first operand)
     * @throws std::runtime_error on unknown set names or malformed expressions
     */
    GroupSet evaluate_expression(const std::string& expr) const {
        // Tokenize: split on '+' and '-', keeping operators
        std::vector<std::string> tokens;
        std::vector<char> ops;  // operators between tokens

        std::string current;
        for (char ch : expr) {
            if (ch == '+' || ch == '-') {
                trim(current);
                if (!current.empty()) tokens.push_back(current);
                ops.push_back(ch);
                current.clear();
            } else {
                current += ch;
            }
        }
        trim(current);
        if (!current.empty()) tokens.push_back(current);

        if (tokens.empty()) {
            throw std::runtime_error("GroupSetManager: empty expression");
        }

        // Bootstrap result with first token
        GroupSet result = get_set(tokens[0]);

        // Apply remaining ops
        for (std::size_t i = 0; i < ops.size(); ++i) {
            if (i + 1 >= tokens.size()) {
                throw std::runtime_error(
                    "GroupSetManager: trailing operator in expression");
            }
            const GroupSet& rhs = get_set(tokens[i + 1]);
            if (ops[i] == '+') {
                result = result.union_with(rhs);
            } else {
                result = result.difference(rhs);
            }
        }

        return result;
    }

private:
    /// Trim whitespace in-place
    static void trim(std::string& s) {
        // leading
        s.erase(s.begin(),
                std::find_if(s.begin(), s.end(),
                             [](unsigned char c){ return !std::isspace(c); }));
        // trailing
        s.erase(std::find_if(s.rbegin(), s.rend(),
                             [](unsigned char c){ return !std::isspace(c); }).base(),
                s.end());
    }

    std::map<std::string, GroupSet> sets_;
};

} // namespace fem
} // namespace nxs
