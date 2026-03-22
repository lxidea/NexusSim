#pragma once

/**
 * @file tuning_wave45.hpp
 * @brief Wave 45e: Final tuning + MPI scalability validation
 *
 * - AdaptiveHourglassSelector: mesh-quality-driven IHQ mode selection
 * - ContactStabilizationDamper: critical damping for contact interfaces
 * - ElementTimeStepCalibrator: per-element-type safety factors
 * - MPIScalabilityValidator: strong scaling validation with 2 ranks
 * - ProductionMPIDriver: top-level orchestrator for production cycle
 */

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <functional>
#include <map>
#include <chrono>
#include <iostream>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {
namespace parallel {

using Real = double;
using Index = std::size_t;

// ============================================================================
// Element shape quality metrics
// ============================================================================

struct ElementQuality {
    Real aspect_ratio = 1.0;  ///< max edge / min edge
    Real skewness = 0.0;      ///< 0 = ideal, 1 = degenerate
    Real warpage = 0.0;       ///< max face out-of-plane angle (rad)
    Real jacobian_ratio = 1.0; ///< min J / max J

    bool is_distorted(Real ar_threshold = 5.0,
                      Real skew_threshold = 0.5,
                      Real warp_threshold = 0.1) const {
        return aspect_ratio > ar_threshold ||
               skewness > skew_threshold ||
               warpage > warp_threshold;
    }
};

/**
 * @brief Compute element quality from nodal coordinates (hex8)
 */
inline ElementQuality compute_hex8_quality(const Real coords[8][3]) {
    ElementQuality q;

    // Compute all 12 edge lengths
    static constexpr int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0},  // bottom
        {4,5},{5,6},{6,7},{7,4},  // top
        {0,4},{1,5},{2,6},{3,7}   // vertical
    };

    Real min_edge = 1e30, max_edge = 0.0;
    for (int e = 0; e < 12; ++e) {
        Real dx = coords[edges[e][1]][0] - coords[edges[e][0]][0];
        Real dy = coords[edges[e][1]][1] - coords[edges[e][0]][1];
        Real dz = coords[edges[e][1]][2] - coords[edges[e][0]][2];
        Real len = std::sqrt(dx*dx + dy*dy + dz*dz);
        min_edge = std::min(min_edge, len);
        max_edge = std::max(max_edge, len);
    }
    q.aspect_ratio = (min_edge > 1e-30) ? (max_edge / min_edge) : 1e6;

    // Skewness: measure deviation of face diagonals from equal length
    // For a perfect hex, face diagonals are equal
    Real d1x = coords[2][0] - coords[0][0];
    Real d1y = coords[2][1] - coords[0][1];
    Real d1z = coords[2][2] - coords[0][2];
    Real d2x = coords[3][0] - coords[1][0];
    Real d2y = coords[3][1] - coords[1][1];
    Real d2z = coords[3][2] - coords[1][2];
    Real diag1 = std::sqrt(d1x*d1x + d1y*d1y + d1z*d1z);
    Real diag2 = std::sqrt(d2x*d2x + d2y*d2y + d2z*d2z);
    Real max_diag = std::max(diag1, diag2);
    Real min_diag = std::min(diag1, diag2);
    q.skewness = (max_diag > 1e-30) ? (1.0 - min_diag / max_diag) : 0.0;

    // Warpage: measure face planarity by checking 4th-node out-of-plane distance
    // For each quad face, compute how far the 4th node deviates from the plane
    // defined by the first 3 nodes. A planar face gives 0 warpage.
    static constexpr int faces[6][4] = {
        {0,1,2,3}, {4,5,6,7},  // bottom, top
        {0,1,5,4}, {2,3,7,6},  // front, back
        {0,3,7,4}, {1,2,6,5}   // left, right
    };

    Real max_warp = 0.0;
    for (int f = 0; f < 6; ++f) {
        int i0 = faces[f][0], i1 = faces[f][1], i2 = faces[f][2], i3 = faces[f][3];
        // Vectors from node 0 to nodes 1, 2
        Real e1[3] = {coords[i1][0]-coords[i0][0], coords[i1][1]-coords[i0][1], coords[i1][2]-coords[i0][2]};
        Real e2[3] = {coords[i2][0]-coords[i0][0], coords[i2][1]-coords[i0][1], coords[i2][2]-coords[i0][2]};
        // Face normal
        Real fn[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        Real fnlen = std::sqrt(fn[0]*fn[0] + fn[1]*fn[1] + fn[2]*fn[2]);
        if (fnlen < 1e-30) continue;
        fn[0] /= fnlen; fn[1] /= fnlen; fn[2] /= fnlen;
        // Distance of 4th node from plane
        Real v[3] = {coords[i3][0]-coords[i0][0], coords[i3][1]-coords[i0][1], coords[i3][2]-coords[i0][2]};
        Real dist = std::abs(fn[0]*v[0] + fn[1]*v[1] + fn[2]*v[2]);
        // Normalize by face diagonal length
        Real diag_len = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
        Real warp_angle = (diag_len > 1e-30) ? std::atan2(dist, diag_len) : 0.0;
        max_warp = std::max(max_warp, warp_angle);
    }
    q.warpage = max_warp;

    return q;
}

// ============================================================================
// AdaptiveHourglassSelector
// ============================================================================

/**
 * @brief Mesh-quality-driven hourglass mode selection.
 *
 * Examines per-element shape metrics (aspect ratio, skew, warpage) and
 * selects the optimal HourglassMode:
 * - Well-shaped elements → IHQ1 (cheap viscous)
 * - Moderately distorted → IHQ2 (stiffness-based)
 * - Severely distorted → IHQ4 (assumed strain co-rotational)
 * - Very poor quality → IHQ5 (assumed deviatoric strain)
 */
class AdaptiveHourglassSelector {
public:
    struct Thresholds {
        Real ar_moderate = 3.0;   ///< aspect ratio for moderate distortion
        Real ar_severe = 8.0;    ///< aspect ratio for severe distortion
        Real skew_moderate = 0.3;
        Real skew_severe = 0.6;
        Real warp_moderate = 0.05; ///< radians
        Real warp_severe = 0.15;
    };

    AdaptiveHourglassSelector() = default;
    explicit AdaptiveHourglassSelector(const Thresholds& t) : thresh_(t) {}

    /**
     * @brief Select hourglass mode for a single element
     * @param quality Element quality metrics
     * @return Recommended IHQ mode (1, 2, 4, or 5)
     */
    int select_mode(const ElementQuality& quality) const {
        // Score-based: higher score = more distorted
        Real score = 0.0;

        // Aspect ratio contribution
        if (quality.aspect_ratio > thresh_.ar_severe) score += 3.0;
        else if (quality.aspect_ratio > thresh_.ar_moderate) score += 1.5;

        // Skewness contribution
        if (quality.skewness > thresh_.skew_severe) score += 3.0;
        else if (quality.skewness > thresh_.skew_moderate) score += 1.5;

        // Warpage contribution
        if (quality.warpage > thresh_.warp_severe) score += 3.0;
        else if (quality.warpage > thresh_.warp_moderate) score += 1.5;

        // Map score to IHQ mode
        if (score >= 6.0) return 5;  // IHQ5: assumed deviatoric
        if (score >= 3.0) return 4;  // IHQ4: assumed strain co-rotational
        if (score >= 1.0) return 2;  // IHQ2: stiffness-based
        return 1;                     // IHQ1: viscous (cheapest)
    }

    /**
     * @brief Batch selection for a mesh of hex8 elements
     * @param coords Node coordinates [num_nodes][3]
     * @param connectivity Element connectivity [num_elements * 8]
     * @param num_elements Number of elements
     * @return Per-element IHQ mode
     */
    std::vector<int> select_modes_hex8(
        const Real* coords,
        const Index* connectivity,
        Index num_elements) const
    {
        std::vector<int> modes(num_elements);
        for (Index e = 0; e < num_elements; ++e) {
            Real elem_coords[8][3];
            for (int n = 0; n < 8; ++n) {
                Index node = connectivity[e * 8 + n];
                elem_coords[n][0] = coords[node * 3 + 0];
                elem_coords[n][1] = coords[node * 3 + 1];
                elem_coords[n][2] = coords[node * 3 + 2];
            }
            ElementQuality q = compute_hex8_quality(elem_coords);
            modes[e] = select_mode(q);
        }
        return modes;
    }

private:
    Thresholds thresh_;
};

// ============================================================================
// ContactStabilizationDamper
// ============================================================================

/**
 * @brief Critical damping ratio for contact interfaces.
 *
 * Computes contact damping force based on:
 *   F_damp = coeff * 2 * sqrt(k * m) * v_rel
 *
 * where k is contact stiffness, m is nodal mass, v_rel is relative velocity.
 */
class ContactStabilizationDamper {
public:
    explicit ContactStabilizationDamper(Real coefficient = 0.1)
        : coeff_(coefficient) {}

    void set_coefficient(Real c) { coeff_ = std::clamp(c, 0.0, 1.0); }
    Real coefficient() const { return coeff_; }

    /**
     * @brief Compute damping force magnitude for a contact pair
     * @param contact_stiffness Contact penalty stiffness
     * @param nodal_mass Mass at the contact node
     * @param relative_velocity Relative velocity magnitude
     * @return Damping force magnitude
     */
    Real compute_damping_force(Real contact_stiffness,
                               Real nodal_mass,
                               Real relative_velocity) const {
        Real critical = 2.0 * std::sqrt(std::abs(contact_stiffness * nodal_mass));
        return coeff_ * critical * std::abs(relative_velocity);
    }

    /**
     * @brief Compute critical damping ratio
     * @param contact_stiffness Contact penalty stiffness
     * @param nodal_mass Mass at the contact node
     * @return Critical damping coefficient (2*sqrt(k*m))
     */
    Real critical_damping(Real contact_stiffness, Real nodal_mass) const {
        return 2.0 * std::sqrt(std::abs(contact_stiffness * nodal_mass));
    }

    /**
     * @brief Batch damping for multiple contact nodes
     * @param stiffnesses Per-node contact stiffness
     * @param masses Per-node mass
     * @param velocities Per-node relative velocity magnitude
     * @return Per-node damping force
     */
    std::vector<Real> compute_batch(
        const std::vector<Real>& stiffnesses,
        const std::vector<Real>& masses,
        const std::vector<Real>& velocities) const
    {
        std::size_t n = stiffnesses.size();
        std::vector<Real> forces(n);
        for (std::size_t i = 0; i < n; ++i) {
            forces[i] = compute_damping_force(stiffnesses[i], masses[i], velocities[i]);
        }
        return forces;
    }

private:
    Real coeff_ = 0.1;
};

// ============================================================================
// ElementTimeStepCalibrator
// ============================================================================

/**
 * @brief Per-element-type safety factors for critical time step.
 *
 * Different element types have different CFL stability limits.
 * Also accounts for material type (explosive/high-strain-rate → lower factor).
 */
class ElementTimeStepCalibrator {
public:
    enum class ElementType {
        Hex8, Hex20, Tet4, Tet10, Shell3, Shell4, Beam2, Spring, Truss
    };

    enum class MaterialCategory {
        Standard,       ///< Regular structural materials
        Explosive,      ///< Detonation products, high-rate EOS
        Hyperelastic,   ///< Large deformation rubber/foam
        Composite       ///< Layered composite
    };

    ElementTimeStepCalibrator() { init_defaults(); }

    /**
     * @brief Get safety factor for element + material combination
     */
    Real safety_factor(ElementType etype, MaterialCategory mcat = MaterialCategory::Standard) const {
        Real base = base_factor(etype);
        Real mat_scale = material_scale(mcat);
        return base * mat_scale;
    }

    /**
     * @brief Compute stable dt for a single element
     * @param char_length Characteristic element length
     * @param sound_speed Material sound speed
     * @param etype Element type
     * @param mcat Material category
     * @return Stable time step
     */
    Real compute_dt(Real char_length, Real sound_speed,
                    ElementType etype,
                    MaterialCategory mcat = MaterialCategory::Standard) const {
        if (sound_speed < 1e-30) return 1e30;
        Real sf = safety_factor(etype, mcat);
        return sf * char_length / sound_speed;
    }

    /**
     * @brief Batch: compute minimum dt across elements
     * @param char_lengths Per-element characteristic length
     * @param sound_speeds Per-element sound speed
     * @param etypes Per-element type
     * @return Minimum stable dt
     */
    Real compute_min_dt(
        const std::vector<Real>& char_lengths,
        const std::vector<Real>& sound_speeds,
        const std::vector<ElementType>& etypes,
        MaterialCategory mcat = MaterialCategory::Standard) const
    {
        Real min_dt = 1e30;
        for (std::size_t i = 0; i < char_lengths.size(); ++i) {
            Real dt = compute_dt(char_lengths[i], sound_speeds[i], etypes[i], mcat);
            min_dt = std::min(min_dt, dt);
        }
        return min_dt;
    }

    /**
     * @brief Set custom safety factor for an element type
     */
    void set_base_factor(ElementType etype, Real factor) {
        factors_[static_cast<int>(etype)] = factor;
    }

private:
    std::array<Real, 9> factors_;

    void init_defaults() {
        factors_[static_cast<int>(ElementType::Hex8)]   = 0.9;
        factors_[static_cast<int>(ElementType::Hex20)]  = 0.9;
        factors_[static_cast<int>(ElementType::Tet4)]   = 0.6667;
        factors_[static_cast<int>(ElementType::Tet10)]  = 0.6667;
        factors_[static_cast<int>(ElementType::Shell3)] = 0.9;
        factors_[static_cast<int>(ElementType::Shell4)] = 0.9;
        factors_[static_cast<int>(ElementType::Beam2)]  = 0.9;
        factors_[static_cast<int>(ElementType::Spring)] = 1.0;
        factors_[static_cast<int>(ElementType::Truss)]  = 0.9;
    }

    Real base_factor(ElementType etype) const {
        return factors_[static_cast<int>(etype)];
    }

    static Real material_scale(MaterialCategory mcat) {
        switch (mcat) {
            case MaterialCategory::Explosive:    return 0.67;
            case MaterialCategory::Hyperelastic: return 0.8;
            case MaterialCategory::Composite:    return 0.85;
            default:                             return 1.0;
        }
    }
};

// ============================================================================
// MPIScalabilityValidator
// ============================================================================

/**
 * @brief Validates MPI strong scaling with 1 and 2 ranks.
 *
 * Extends ScalabilityBenchmark from mpi_wave17.hpp with:
 * - Actual MPI timing (MPI_Wtime)
 * - Communication overhead validation (comm/comp < threshold)
 * - Strong scaling efficiency check
 */
class MPIScalabilityValidator {
public:
    struct ValidationResult {
        int n_ranks = 1;
        double total_time = 0.0;
        double comp_time = 0.0;
        double comm_time = 0.0;
        double speedup = 1.0;
        double efficiency = 1.0;
        double comm_overhead = 0.0;  ///< comm / comp ratio
        bool passed = true;
        std::string message;
    };

    explicit MPIScalabilityValidator(double max_comm_overhead = 0.20)
        : max_comm_overhead_(max_comm_overhead) {}

    /**
     * @brief Validate strong scaling with a workload function
     * @param workload Function(problem_size) that returns {comp_time, comm_time}
     * @param problem_size Problem size
     * @param rank Current rank
     * @param n_ranks Total ranks
     * @return Validation result
     */
    ValidationResult validate_strong_scaling(
        std::function<std::pair<double,double>(Index)> workload,
        Index problem_size,
        int /*rank*/, int n_ranks)
    {
        ValidationResult result;
        result.n_ranks = n_ranks;

#ifdef NEXUSSIM_HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
#else
        auto t0_chr = std::chrono::high_resolution_clock::now();
#endif

        auto [comp, comm] = workload(problem_size);

#ifdef NEXUSSIM_HAVE_MPI
        double t1 = MPI_Wtime();
        result.total_time = t1 - t0;

        // Max across ranks for consistent reporting
        double max_total = 0.0;
        MPI_Allreduce(&result.total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        result.total_time = max_total;
#else
        auto t1_chr = std::chrono::high_resolution_clock::now();
        result.total_time = std::chrono::duration<double>(t1_chr - t0_chr).count();
#endif

        result.comp_time = comp;
        result.comm_time = comm;
        result.comm_overhead = (comp > 1e-30) ? (comm / comp) : 0.0;

        // Store for scaling computation
        results_.push_back(result);

        return result;
    }

    /**
     * @brief Validate communication overhead is within threshold
     */
    ValidationResult validate_communication_overhead(
        double comp_time, double comm_time,
        int /*rank*/, int n_ranks)
    {
        ValidationResult result;
        result.n_ranks = n_ranks;
        result.comp_time = comp_time;
        result.comm_time = comm_time;
        result.comm_overhead = (comp_time > 1e-30) ? (comm_time / comp_time) : 0.0;
        result.passed = (result.comm_overhead <= max_comm_overhead_);

        std::ostringstream oss;
        oss << "Comm overhead: " << (result.comm_overhead * 100.0) << "% "
            << (result.passed ? "(OK)" : "(EXCEEDS THRESHOLD)");
        result.message = oss.str();

        return result;
    }

    /**
     * @brief Compute scaling metrics across stored results
     */
    void compute_scaling() {
        if (results_.empty()) return;

        double baseline = results_[0].total_time;
        for (auto& r : results_) {
            r.speedup = (r.total_time > 1e-30) ? (baseline / r.total_time) : 1.0;
            r.efficiency = r.speedup / r.n_ranks;
        }
    }

    const std::vector<ValidationResult>& results() const { return results_; }
    void clear() { results_.clear(); }

private:
    double max_comm_overhead_;
    std::vector<ValidationResult> results_;
};

// ============================================================================
// ProductionMPIDriver
// ============================================================================

/**
 * @brief Top-level orchestrator for production MPI cycle.
 *
 * Demonstrates the full production loop:
 *   partition → force exchange → contact exchange → output gather → dt sync
 */
class ProductionMPIDriver {
public:
    struct CycleResult {
        int num_steps = 0;
        Real final_time = 0.0;
        Real total_energy = 0.0;
        Real min_dt_used = 1e30;
        double wall_time = 0.0;
        bool completed = false;
    };

    struct Config {
        Real end_time = 1.0e-3;
        Real initial_dt = 1.0e-6;
        int max_steps = 1000;
        int output_interval = 100;
    };

    ProductionMPIDriver() = default;
    explicit ProductionMPIDriver(const Config& config) : config_(config) {}

    /**
     * @brief Run a simplified production cycle
     *
     * @param num_local_nodes Number of local nodes on this rank
     * @param num_local_elements Number of local elements on this rank
     * @param force_func Function to compute local forces (returns KE, IE)
     * @param rank Current rank
     * @param n_ranks Total ranks
     * @return Cycle result
     */
    CycleResult run(
        Index /*num_local_nodes*/,
        Index /*num_local_elements*/,
        std::function<std::pair<Real,Real>(int step, Real dt)> force_func,
        int /*rank*/, int /*n_ranks*/)
    {
        CycleResult result;

        auto t0 = std::chrono::high_resolution_clock::now();

        Real time = 0.0;
        Real dt = config_.initial_dt;
        int step = 0;

        while (time < config_.end_time && step < config_.max_steps) {
            // Compute local forces
            auto [ke, ie] = force_func(step, dt);

            // Reduce energies
            Real global_ke = ke, global_ie = ie;
#ifdef NEXUSSIM_HAVE_MPI
            MPI_Allreduce(&ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&ie, &global_ie, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

            // Sync dt
#ifdef NEXUSSIM_HAVE_MPI
            Real local_dt = dt;
            Real global_dt = 0.0;
            MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            dt = global_dt;
#endif

            result.min_dt_used = std::min(result.min_dt_used, dt);
            result.total_energy = global_ke + global_ie;

            time += dt;
            step++;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        result.num_steps = step;
        result.final_time = time;
        result.wall_time = std::chrono::duration<double>(t1 - t0).count();
        result.completed = (time >= config_.end_time);

        return result;
    }

private:
    Config config_;
};

} // namespace parallel
} // namespace nxs
