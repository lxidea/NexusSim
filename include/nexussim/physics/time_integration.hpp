#pragma once

/**
 * @file time_integration.hpp
 * @brief Advanced time integration schemes with subcycling and energy monitoring
 *
 * Features:
 * - Subcycling for multi-scale problems (different dt for different regions)
 * - Consistent mass matrix support
 * - Energy conservation monitoring
 * - Multiple integration schemes (Central Difference, Newmark-β)
 *
 * Subcycling Algorithm:
 * - Partition mesh into fast/slow regions based on CFL
 * - Fast region takes N substeps per slow region step
 * - Interface nodes updated with proper synchronization
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/adaptive_timestep.hpp>
#include <Kokkos_Core.hpp>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace nxs {
namespace physics {

// ============================================================================
// Time Integration Schemes
// ============================================================================

enum class IntegrationScheme {
    CentralDifference,   ///< Explicit central difference (default)
    NewmarkExplicit,     ///< Newmark-β with β=0, γ=0.5 (explicit)
    NewmarkImplicit,     ///< Newmark-β with β=0.25, γ=0.5 (implicit, unconditionally stable)
    HHT_Alpha,           ///< Hilber-Hughes-Taylor α-method (numerical damping)
    VelocityVerlet       ///< Symplectic velocity Verlet (energy conserving)
};

// ============================================================================
// Energy Monitor
// ============================================================================

/**
 * @brief Tracks energy conservation during simulation
 */
class EnergyMonitor {
public:
    struct EnergyState {
        Real kinetic = 0.0;       ///< Kinetic energy
        Real internal = 0.0;      ///< Internal (strain) energy
        Real external_work = 0.0; ///< Work done by external forces
        Real contact_work = 0.0;  ///< Work done by contact forces
        Real hourglass = 0.0;     ///< Hourglass energy (artificial)
        Real damping = 0.0;       ///< Energy dissipated by damping
        Real time = 0.0;          ///< Simulation time

        Real total() const {
            return kinetic + internal;
        }

        Real balance() const {
            // Energy balance: E_total = E_initial + W_external - W_damping
            return total() - external_work + damping - contact_work;
        }
    };

    EnergyMonitor() = default;

    void record(const EnergyState& state) {
        history_.push_back(state);

        if (history_.size() == 1) {
            initial_energy_ = state.total();
        }

        current_ = state;
    }

    Real relative_error() const {
        if (std::abs(initial_energy_) < 1e-20) return 0.0;
        return std::abs(current_.balance() - initial_energy_) / initial_energy_;
    }

    bool is_conserved(Real tolerance = 0.01) const {
        return relative_error() < tolerance;
    }

    const EnergyState& current() const { return current_; }
    const std::vector<EnergyState>& history() const { return history_; }

    void print_stats(std::ostream& os = std::cout) const {
        os << "=== Energy Monitor ===\n";
        os << "Kinetic:      " << current_.kinetic << " J\n";
        os << "Internal:     " << current_.internal << " J\n";
        os << "Total:        " << current_.total() << " J\n";
        os << "External work:" << current_.external_work << " J\n";
        os << "Initial:      " << initial_energy_ << " J\n";
        os << "Rel. error:   " << (relative_error() * 100) << " %\n";
        os << "=====================\n";
    }

    void reset() {
        history_.clear();
        current_ = EnergyState();
        initial_energy_ = 0.0;
    }

private:
    std::vector<EnergyState> history_;
    EnergyState current_;
    Real initial_energy_ = 0.0;
};

// ============================================================================
// Subcycling Region
// ============================================================================

/**
 * @brief Defines a region with its own timestep for subcycling
 */
struct SubcycleRegion {
    std::string name;
    std::vector<Index> node_ids;      ///< Nodes in this region
    std::vector<Index> element_ids;   ///< Elements in this region
    std::vector<Index> interface_nodes; ///< Nodes shared with other regions
    Real dt_scale = 1.0;              ///< Timestep scale relative to master dt
    int subcycle_ratio = 1;           ///< Number of substeps per master step

    SubcycleRegion(const std::string& n = "default")
        : name(n) {}
};

// ============================================================================
// Subcycling Controller
// ============================================================================

/**
 * @brief Manages subcycling for multi-scale time integration
 *
 * Subcycling allows different parts of the mesh to use different timesteps:
 * - Fine elements (high frequency) take multiple small steps
 * - Coarse elements (low frequency) take fewer large steps
 * - Interface nodes are synchronized between regions
 *
 * Usage:
 * ```cpp
 * SubcyclingController subcycle;
 * subcycle.add_region("fine", fine_nodes, fine_elements, 4);  // 4x substeps
 * subcycle.add_region("coarse", coarse_nodes, coarse_elements, 1);
 * subcycle.compute_interface_nodes();
 *
 * // In time loop:
 * Real master_dt = subcycle.compute_master_dt(cfl_dts);
 * subcycle.step(master_dt, [&](const SubcycleRegion& region, Real dt) {
 *     // Advance region by dt
 *     solver.step_region(region, dt);
 * });
 * ```
 */
class SubcyclingController {
public:
    SubcyclingController() = default;

    /**
     * @brief Add a subcycling region
     * @param name Region name
     * @param nodes Node IDs in region
     * @param elements Element IDs in region
     * @param subcycle_ratio Number of substeps per master step
     */
    void add_region(const std::string& name,
                    const std::vector<Index>& nodes,
                    const std::vector<Index>& elements,
                    int subcycle_ratio = 1) {
        SubcycleRegion region(name);
        region.node_ids = nodes;
        region.element_ids = elements;
        region.subcycle_ratio = subcycle_ratio;
        regions_.push_back(region);
    }

    /**
     * @brief Automatically partition mesh based on element stable dt
     * @param element_dts Stable dt for each element
     * @param ratio_threshold Ratio above which to create separate region
     */
    void auto_partition(const std::vector<Real>& element_dts,
                        Real ratio_threshold = 4.0) {
        if (element_dts.empty()) return;

        // Find min/max dt
        Real dt_min = *std::min_element(element_dts.begin(), element_dts.end());
        Real dt_max = *std::max_element(element_dts.begin(), element_dts.end());

        if (dt_max / dt_min < ratio_threshold) {
            // No need for subcycling - single region
            SubcycleRegion region("uniform");
            for (size_t i = 0; i < element_dts.size(); ++i) {
                region.element_ids.push_back(i);
            }
            region.subcycle_ratio = 1;
            regions_.push_back(region);
            return;
        }

        // Partition into fast/slow regions
        Real dt_threshold = dt_min * std::sqrt(ratio_threshold);

        SubcycleRegion fast("fast"), slow("slow");
        for (size_t i = 0; i < element_dts.size(); ++i) {
            if (element_dts[i] < dt_threshold) {
                fast.element_ids.push_back(i);
            } else {
                slow.element_ids.push_back(i);
            }
        }

        // Compute subcycle ratios
        Real fast_dt = dt_min;
        Real slow_dt = dt_max;
        int ratio = static_cast<int>(std::ceil(slow_dt / fast_dt));

        fast.subcycle_ratio = ratio;
        slow.subcycle_ratio = 1;

        if (!fast.element_ids.empty()) regions_.push_back(fast);
        if (!slow.element_ids.empty()) regions_.push_back(slow);
    }

    /**
     * @brief Compute interface nodes (shared between regions)
     * @param element_connectivity Element to node connectivity
     */
    void compute_interface_nodes(
        const std::function<std::vector<Index>(Index)>& element_connectivity) {

        if (regions_.size() < 2) return;

        // Build node-to-region map
        std::map<Index, std::set<size_t>> node_regions;

        for (size_t r = 0; r < regions_.size(); ++r) {
            for (Index elem : regions_[r].element_ids) {
                auto nodes = element_connectivity(elem);
                for (Index node : nodes) {
                    node_regions[node].insert(r);
                    // Also add to region's node list
                    auto& region_nodes = regions_[r].node_ids;
                    if (std::find(region_nodes.begin(), region_nodes.end(), node)
                        == region_nodes.end()) {
                        region_nodes.push_back(node);
                    }
                }
            }
        }

        // Find interface nodes (belong to multiple regions)
        for (const auto& [node, region_set] : node_regions) {
            if (region_set.size() > 1) {
                for (size_t r : region_set) {
                    regions_[r].interface_nodes.push_back(node);
                }
            }
        }
    }

    /**
     * @brief Compute master timestep
     * @param region_dts Stable dt for each region
     * @return Master timestep (largest region dt)
     */
    Real compute_master_dt(const std::vector<Real>& region_dts) const {
        if (region_dts.empty()) return 0.0;

        Real master_dt = 0.0;
        for (size_t i = 0; i < regions_.size() && i < region_dts.size(); ++i) {
            Real effective_dt = region_dts[i] * regions_[i].subcycle_ratio;
            master_dt = std::max(master_dt, effective_dt);
        }
        return master_dt;
    }

    /**
     * @brief Execute one master timestep with subcycling
     * @param master_dt Master timestep
     * @param step_func Function to advance a region by dt
     */
    void step(Real master_dt,
              const std::function<void(const SubcycleRegion&, Real, int)>& step_func) {

        // For each region, take appropriate number of substeps
        for (const auto& region : regions_) {
            Real sub_dt = master_dt / region.subcycle_ratio;

            for (int substep = 0; substep < region.subcycle_ratio; ++substep) {
                step_func(region, sub_dt, substep);
            }
        }

        // Synchronize interface nodes (average velocities)
        // This would be done by the step_func or externally
    }

    /**
     * @brief Get all regions
     */
    const std::vector<SubcycleRegion>& regions() const { return regions_; }
    std::vector<SubcycleRegion>& regions() { return regions_; }

    size_t num_regions() const { return regions_.size(); }

    bool has_subcycling() const {
        for (const auto& r : regions_) {
            if (r.subcycle_ratio > 1) return true;
        }
        return false;
    }

    void print_info(std::ostream& os = std::cout) const {
        os << "=== Subcycling Info ===\n";
        os << "Regions: " << regions_.size() << "\n";
        for (const auto& r : regions_) {
            os << "  " << r.name << ": "
               << r.element_ids.size() << " elements, "
               << r.node_ids.size() << " nodes, "
               << r.interface_nodes.size() << " interface, "
               << "ratio=" << r.subcycle_ratio << "\n";
        }
        os << "=======================\n";
    }

private:
    std::vector<SubcycleRegion> regions_;
};

// ============================================================================
// Consistent Mass Matrix
// ============================================================================

/**
 * @brief Consistent mass matrix computation and storage
 *
 * Unlike lumped mass (diagonal), consistent mass preserves higher accuracy
 * for wave propagation but requires solving a system.
 */
class ConsistentMass {
public:
    /**
     * @brief Mass matrix entry (sparse storage)
     */
    struct Entry {
        Index row;
        Index col;
        Real value;
    };

    ConsistentMass() = default;

    /**
     * @brief Add mass matrix contribution
     */
    void add_entry(Index i, Index j, Real m) {
        entries_.push_back({i, j, m});
        if (i != j) {
            entries_.push_back({j, i, m});  // Symmetric
        }
    }

    /**
     * @brief Build CSR format for efficient SpMV
     */
    void build_csr(size_t num_dofs) {
        num_dofs_ = num_dofs;

        // Sort entries by row, then column
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry& a, const Entry& b) {
                      return (a.row < b.row) || (a.row == b.row && a.col < b.col);
                  });

        // Merge duplicates
        std::vector<Entry> merged;
        for (const auto& e : entries_) {
            if (!merged.empty() && merged.back().row == e.row && merged.back().col == e.col) {
                merged.back().value += e.value;
            } else {
                merged.push_back(e);
            }
        }
        entries_ = std::move(merged);

        // Build CSR
        row_ptr_.resize(num_dofs + 1, 0);
        col_idx_.resize(entries_.size());
        values_.resize(entries_.size());

        for (size_t i = 0; i < entries_.size(); ++i) {
            row_ptr_[entries_[i].row + 1]++;
            col_idx_[i] = entries_[i].col;
            values_[i] = entries_[i].value;
        }

        // Cumulative sum for row_ptr
        for (size_t i = 1; i <= num_dofs; ++i) {
            row_ptr_[i] += row_ptr_[i - 1];
        }
    }

    /**
     * @brief Sparse matrix-vector multiply: y = M * x
     */
    void multiply(const Real* x, Real* y) const {
        for (size_t i = 0; i < num_dofs_; ++i) {
            y[i] = 0.0;
            for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
                y[i] += values_[j] * x[col_idx_[j]];
            }
        }
    }

    /**
     * @brief Solve M * x = b using Jacobi iteration
     * @param b Right-hand side
     * @param x Solution (in/out, should be initialized)
     * @param tol Convergence tolerance
     * @param max_iter Maximum iterations
     * @return Number of iterations
     */
    int solve_jacobi(const Real* b, Real* x, Real tol = 1e-6, int max_iter = 100) const {
        std::vector<Real> x_new(num_dofs_);
        std::vector<Real> diag(num_dofs_, 0.0);

        // Extract diagonal
        for (size_t i = 0; i < num_dofs_; ++i) {
            for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
                if (col_idx_[j] == i) {
                    diag[i] = values_[j];
                    break;
                }
            }
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            Real max_diff = 0.0;

            for (size_t i = 0; i < num_dofs_; ++i) {
                Real sum = b[i];
                for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
                    if (col_idx_[j] != i) {
                        sum -= values_[j] * x[col_idx_[j]];
                    }
                }
                x_new[i] = sum / diag[i];
                max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
            }

            std::copy(x_new.begin(), x_new.end(), x);

            if (max_diff < tol) {
                return iter + 1;
            }
        }

        return max_iter;
    }

    size_t num_dofs() const { return num_dofs_; }
    size_t nnz() const { return entries_.size(); }

    void clear() {
        entries_.clear();
        row_ptr_.clear();
        col_idx_.clear();
        values_.clear();
        num_dofs_ = 0;
    }

private:
    std::vector<Entry> entries_;
    std::vector<size_t> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Real> values_;
    size_t num_dofs_ = 0;
};

// ============================================================================
// Advanced Time Integrator
// ============================================================================

/**
 * @brief Advanced time integrator with subcycling and energy monitoring
 */
class AdvancedTimeIntegrator {
public:
    AdvancedTimeIntegrator(IntegrationScheme scheme = IntegrationScheme::CentralDifference)
        : scheme_(scheme)
        , use_consistent_mass_(false)
        , damping_alpha_(0.0)  // Rayleigh mass damping
        , damping_beta_(0.0)   // Rayleigh stiffness damping
    {}

    // ========================================================================
    // Configuration
    // ========================================================================

    void set_scheme(IntegrationScheme scheme) { scheme_ = scheme; }
    IntegrationScheme scheme() const { return scheme_; }

    void enable_consistent_mass(bool enable) { use_consistent_mass_ = enable; }
    bool uses_consistent_mass() const { return use_consistent_mass_; }

    void set_rayleigh_damping(Real alpha, Real beta) {
        damping_alpha_ = alpha;
        damping_beta_ = beta;
    }

    // Newmark parameters
    void set_newmark_params(Real beta, Real gamma) {
        newmark_beta_ = beta;
        newmark_gamma_ = gamma;
    }

    // HHT-α parameter
    void set_hht_alpha(Real alpha) {
        hht_alpha_ = alpha;
        // Optimal Newmark params for HHT
        newmark_beta_ = (1.0 - alpha) * (1.0 - alpha) / 4.0;
        newmark_gamma_ = 0.5 - alpha;
    }

    // ========================================================================
    // Central Difference (Explicit)
    // ========================================================================

    /**
     * @brief Central difference velocity update
     * v^{n+1/2} = v^{n-1/2} + dt * a^n
     */
    KOKKOS_INLINE_FUNCTION
    static void central_diff_velocity(Real* v_half, const Real* v_half_old,
                                       const Real* a, Real dt, size_t ndof) {
        for (size_t i = 0; i < ndof; ++i) {
            v_half[i] = v_half_old[i] + dt * a[i];
        }
    }

    /**
     * @brief Central difference displacement update
     * u^{n+1} = u^n + dt * v^{n+1/2}
     */
    KOKKOS_INLINE_FUNCTION
    static void central_diff_displacement(Real* u, const Real* v_half,
                                           Real dt, size_t ndof) {
        for (size_t i = 0; i < ndof; ++i) {
            u[i] += dt * v_half[i];
        }
    }

    // ========================================================================
    // Velocity Verlet (Symplectic)
    // ========================================================================

    /**
     * @brief Velocity Verlet half-step velocity
     * v^{n+1/2} = v^n + (dt/2) * a^n
     */
    KOKKOS_INLINE_FUNCTION
    static void verlet_velocity_half(Real* v_half, const Real* v, const Real* a,
                                      Real dt, size_t ndof) {
        for (size_t i = 0; i < ndof; ++i) {
            v_half[i] = v[i] + 0.5 * dt * a[i];
        }
    }

    /**
     * @brief Velocity Verlet full velocity update
     * v^{n+1} = v^{n+1/2} + (dt/2) * a^{n+1}
     */
    KOKKOS_INLINE_FUNCTION
    static void verlet_velocity_full(Real* v, const Real* v_half, const Real* a_new,
                                      Real dt, size_t ndof) {
        for (size_t i = 0; i < ndof; ++i) {
            v[i] = v_half[i] + 0.5 * dt * a_new[i];
        }
    }

    // ========================================================================
    // Newmark-β
    // ========================================================================

    /**
     * @brief Newmark predictor step
     */
    void newmark_predict(Real* u, Real* v, Real* a,
                         const Real* u_old, const Real* v_old, const Real* a_old,
                         Real dt, size_t ndof) const {
        Real beta = newmark_beta_;
        Real gamma = newmark_gamma_;

        for (size_t i = 0; i < ndof; ++i) {
            // Predict displacement
            u[i] = u_old[i] + dt * v_old[i] +
                   dt * dt * (0.5 - beta) * a_old[i];

            // Predict velocity
            v[i] = v_old[i] + dt * (1.0 - gamma) * a_old[i];

            // Acceleration will be computed from equilibrium
            a[i] = 0.0;
        }
    }

    /**
     * @brief Newmark corrector step (after solving for a^{n+1})
     */
    void newmark_correct(Real* u, Real* v, const Real* a,
                         Real dt, size_t ndof) const {
        Real beta = newmark_beta_;
        Real gamma = newmark_gamma_;

        for (size_t i = 0; i < ndof; ++i) {
            u[i] += beta * dt * dt * a[i];
            v[i] += gamma * dt * a[i];
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    SubcyclingController& subcycling() { return subcycling_; }
    const SubcyclingController& subcycling() const { return subcycling_; }

    EnergyMonitor& energy_monitor() { return energy_monitor_; }
    const EnergyMonitor& energy_monitor() const { return energy_monitor_; }

    ConsistentMass& consistent_mass() { return consistent_mass_; }
    const ConsistentMass& consistent_mass() const { return consistent_mass_; }

    Real damping_alpha() const { return damping_alpha_; }
    Real damping_beta() const { return damping_beta_; }

private:
    IntegrationScheme scheme_;
    bool use_consistent_mass_;

    // Rayleigh damping: C = α*M + β*K
    Real damping_alpha_;
    Real damping_beta_;

    // Newmark parameters
    Real newmark_beta_ = 0.25;   // Average acceleration (implicit)
    Real newmark_gamma_ = 0.5;   // No numerical damping

    // HHT-α parameter
    Real hht_alpha_ = 0.0;       // 0 = Newmark, -1/3 to 0 for damping

    // Subcycling
    SubcyclingController subcycling_;

    // Energy monitoring
    EnergyMonitor energy_monitor_;

    // Consistent mass
    ConsistentMass consistent_mass_;
};

} // namespace physics
} // namespace nxs
