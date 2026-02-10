#pragma once

/**
 * @file pd_fem_coupling.hpp
 * @brief FEM-Peridynamics coupling interface
 *
 * Implements Arlequin-style domain decomposition for FEM-PD coupling:
 * - Overlapping domain with blending functions
 * - Force/displacement transfer at interface
 * - Adaptive FEM-to-PD conversion for damage
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_solver.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <memory>
#include <vector>
#include <map>

namespace nxs {

// Forward declare FEM types
namespace fem {
    class FEMSolver;
}

namespace pd {

// Type alias for FEM solver in PD namespace
using FEMSolver = fem::FEMSolver;

// ============================================================================
// Coupling Region Types
// ============================================================================

/**
 * @brief Domain type for each node/particle
 */
enum class DomainType {
    FEM_Only,       ///< Pure FEM region
    PD_Only,        ///< Pure PD region
    Overlap,        ///< Overlapping region (Arlequin)
    Interface       ///< Sharp interface (constraint-based)
};

/**
 * @brief Coupling method
 */
enum class CouplingMethod {
    Arlequin,       ///< Overlapping domain with energy blending
    MortarContact,  ///< Mortar-based interface coupling
    DirectForce,    ///< Direct force transfer
    Morphing        ///< Adaptive mesh-to-particle morphing
};

// ============================================================================
// Coupling Configuration
// ============================================================================

/**
 * @brief Configuration for FEM-PD coupling
 */
struct FEMPDCouplingConfig {
    CouplingMethod method = CouplingMethod::Arlequin;

    // Arlequin parameters
    Real blend_width = 0.0;         ///< Width of blending zone (auto if 0)
    Real blend_exponent = 2.0;      ///< Blending function exponent

    // Mortar parameters
    Real mortar_tolerance = 1e-6;   ///< Mortar integration tolerance
    Index mortar_quadrature = 3;    ///< Gauss quadrature order

    // Morphing parameters
    Real damage_threshold = 0.3;    ///< Damage threshold for FEM-to-PD conversion
    Real particle_spacing = 0.0;    ///< Particle spacing (auto from mesh if 0)

    // Synchronization
    bool sync_displacement = true;  ///< Sync displacements at interface
    bool sync_velocity = true;      ///< Sync velocities at interface
    bool sync_acceleration = true;  ///< Sync accelerations at interface
};

// ============================================================================
// Coupling Node/Particle Pair
// ============================================================================

/**
 * @brief Mapping between FEM node and PD particle
 */
struct NodeParticleMap {
    Index fem_node_id;          ///< FEM mesh node index
    Index pd_particle_id;       ///< PD particle index
    Real weight;                ///< Coupling weight (for blending)
    DomainType domain;          ///< Domain classification
};

/**
 * @brief Interface segment connecting FEM and PD domains
 */
struct InterfaceSegment {
    std::vector<Index> fem_nodes;       ///< FEM nodes on interface
    std::vector<Index> pd_particles;    ///< PD particles on interface
    Real area;                          ///< Interface area
    Real normal[3];                     ///< Interface normal (FEM to PD)
};

// ============================================================================
// FEM-PD Coupling Manager
// ============================================================================

/**
 * @brief Main coupling manager for FEM-PD simulations
 *
 * Handles:
 * - Domain decomposition and interface detection
 * - Blending function computation
 * - Force and displacement transfer
 * - Adaptive FEM-to-PD conversion
 */
class FEMPDCoupling {
public:
    FEMPDCoupling() = default;

    /**
     * @brief Initialize coupling
     * @param config Coupling configuration
     */
    void initialize(const FEMPDCouplingConfig& config) {
        config_ = config;
        is_initialized_ = true;
        NXS_LOG_INFO("FEMPDCoupling initialized: method={}",
                     static_cast<int>(config_.method));
    }

    /**
     * @brief Set FEM solver
     */
    void set_fem_solver(std::shared_ptr<FEMSolver> fem_solver) {
        fem_solver_ = fem_solver;
    }

    /**
     * @brief Set PD solver
     */
    void set_pd_solver(std::shared_ptr<PDSolver> pd_solver) {
        pd_solver_ = pd_solver;
    }

    /**
     * @brief Build coupling from overlapping regions
     *
     * Detects overlap between FEM mesh and PD particle cloud,
     * builds interface mapping, and computes blending functions.
     */
    void build_coupling() {
        if (!fem_solver_ || !pd_solver_) {
            NXS_LOG_ERROR("FEMPDCoupling: solvers not set");
            return;
        }

        auto& mesh = fem_solver_->mesh();
        auto& particles = pd_solver_->particles();

        Index num_nodes = mesh.num_nodes();
        Index num_particles = particles.num_particles();

        // Sync particle positions to host
        particles.sync_to_host();
        auto x_pd = particles.x_host();

        // Classify each FEM node
        fem_domain_.resize(num_nodes, DomainType::FEM_Only);
        pd_domain_.resize(num_particles, DomainType::PD_Only);

        // Find bounding boxes
        Real fem_min[3] = {1e30, 1e30, 1e30};
        Real fem_max[3] = {-1e30, -1e30, -1e30};
        Real pd_min[3] = {1e30, 1e30, 1e30};
        Real pd_max[3] = {-1e30, -1e30, -1e30};

        for (Index i = 0; i < num_nodes; ++i) {
            Vec3r coords = mesh.get_node_coordinates(i);
            for (int d = 0; d < 3; ++d) {
                fem_min[d] = std::min(fem_min[d], coords[d]);
                fem_max[d] = std::max(fem_max[d], coords[d]);
            }
        }

        for (Index i = 0; i < num_particles; ++i) {
            for (int d = 0; d < 3; ++d) {
                pd_min[d] = std::min(pd_min[d], x_pd(i, d));
                pd_max[d] = std::max(pd_max[d], x_pd(i, d));
            }
        }

        // Compute overlap region
        Real overlap_min[3], overlap_max[3];
        bool has_overlap = true;
        for (int d = 0; d < 3; ++d) {
            overlap_min[d] = std::max(fem_min[d], pd_min[d]);
            overlap_max[d] = std::min(fem_max[d], pd_max[d]);
            if (overlap_min[d] > overlap_max[d]) {
                has_overlap = false;
            }
        }

        if (!has_overlap) {
            NXS_LOG_WARN("FEMPDCoupling: No overlap detected between FEM and PD domains");
            return;
        }

        // Blending zone width (auto-compute if not set)
        Real blend_width = config_.blend_width;
        if (blend_width <= 0.0) {
            // Use 2x average particle horizon as default
            Real avg_horizon = 0.0;
            auto horizon_host = particles.horizon_host();
            for (Index i = 0; i < num_particles; ++i) {
                avg_horizon += horizon_host(i);
            }
            avg_horizon /= num_particles;
            blend_width = 2.0 * avg_horizon;
        }

        // Classify nodes and particles
        node_particle_map_.clear();

        // For each FEM node, check if in overlap and find nearest PD particle
        for (Index i = 0; i < num_nodes; ++i) {
            Vec3r node_coords = mesh.get_node_coordinates(i);
            Real x[3] = {node_coords[0], node_coords[1], node_coords[2]};

            // Check if in overlap region
            bool in_overlap = true;
            for (int d = 0; d < 3; ++d) {
                if (x[d] < overlap_min[d] || x[d] > overlap_max[d]) {
                    in_overlap = false;
                    break;
                }
            }

            if (in_overlap) {
                // Find nearest PD particle
                Index nearest_pd = 0;
                Real min_dist = 1e30;

                for (Index j = 0; j < num_particles; ++j) {
                    Real dx = x[0] - x_pd(j, 0);
                    Real dy = x[1] - x_pd(j, 1);
                    Real dz = x[2] - x_pd(j, 2);
                    Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_pd = j;
                    }
                }

                // Compute blending weight based on position in overlap
                Real alpha = compute_blend_weight(x, overlap_min, overlap_max, blend_width);

                NodeParticleMap mapping;
                mapping.fem_node_id = i;
                mapping.pd_particle_id = nearest_pd;
                mapping.weight = alpha;
                mapping.domain = DomainType::Overlap;

                node_particle_map_.push_back(mapping);
                fem_domain_[i] = DomainType::Overlap;
                pd_domain_[nearest_pd] = DomainType::Overlap;
            }
        }

        NXS_LOG_INFO("FEMPDCoupling: {} coupled node-particle pairs, blend_width={}",
                     node_particle_map_.size(), blend_width);
    }

    /**
     * @brief Transfer displacements from FEM to PD in overlap region
     */
    void transfer_fem_to_pd() {
        if (node_particle_map_.empty()) return;

        auto& mesh = fem_solver_->mesh();
        auto& particles = pd_solver_->particles();

        auto fem_disp = fem_solver_->displacement();
        auto u = particles.u();
        auto v = particles.v();

        // Transfer on device
        auto map_device = Kokkos::View<NodeParticleMap*>("map", node_particle_map_.size());
        auto map_host = Kokkos::create_mirror_view(map_device);
        for (size_t i = 0; i < node_particle_map_.size(); ++i) {
            map_host(i) = node_particle_map_[i];
        }
        Kokkos::deep_copy(map_device, map_host);

        Index num_coupled = node_particle_map_.size();

        Kokkos::parallel_for("fem_to_pd_transfer", num_coupled,
            KOKKOS_LAMBDA(const Index i) {
                auto& m = map_device(i);
                Index fem_id = m.fem_node_id;
                Index pd_id = m.pd_particle_id;
                Real alpha = m.weight;  // FEM weight (1-alpha for PD)

                // Blend: u_pd = alpha * u_fem + (1-alpha) * u_pd
                // Note: FEM displacement is stored as flat array [node*3 + dof]
                for (int d = 0; d < 3; ++d) {
                    u(pd_id, d) = alpha * fem_disp(fem_id * 3 + d) + (1.0 - alpha) * u(pd_id, d);
                }
            });
    }

    /**
     * @brief Transfer forces from PD to FEM in overlap region
     */
    void transfer_pd_to_fem() {
        if (node_particle_map_.empty()) return;

        auto& particles = pd_solver_->particles();
        auto f_pd = particles.f();
        auto volume = particles.volume();

        // For each coupled pair, add PD force contribution to FEM
        // This modifies the FEM external force vector
        auto& f_ext_dual = fem_solver_->external_force();
        f_ext_dual.sync_device();
        auto f_ext = f_ext_dual.view_device();

        auto map_device = Kokkos::View<NodeParticleMap*>("map", node_particle_map_.size());
        auto map_host = Kokkos::create_mirror_view(map_device);
        for (size_t i = 0; i < node_particle_map_.size(); ++i) {
            map_host(i) = node_particle_map_[i];
        }
        Kokkos::deep_copy(map_device, map_host);

        Index num_coupled = node_particle_map_.size();

        Kokkos::parallel_for("pd_to_fem_transfer", num_coupled,
            KOKKOS_LAMBDA(const Index i) {
                auto& m = map_device(i);
                Index fem_id = m.fem_node_id;
                Index pd_id = m.pd_particle_id;
                Real alpha = m.weight;

                // Add PD force contribution (weighted by 1-alpha)
                Real pd_weight = 1.0 - alpha;
                Real V = volume(pd_id);

                // Note: FEM force is stored as flat array [node*3 + dof]
                for (int d = 0; d < 3; ++d) {
                    // f = force_density * volume
                    Kokkos::atomic_add(&f_ext(fem_id * 3 + d), pd_weight * f_pd(pd_id, d) * V);
                }
            });

        f_ext_dual.modify_device();
    }

    /**
     * @brief Synchronize state between FEM and PD at interface
     */
    void synchronize() {
        if (config_.sync_displacement) {
            transfer_fem_to_pd();
        }
        transfer_pd_to_fem();
    }

    /**
     * @brief Convert damaged FEM elements to PD particles
     *
     * When damage in FEM elements exceeds threshold, converts
     * the element to PD particles for fracture simulation.
     *
     * @return Number of elements converted
     */
    Index convert_damaged_elements() {
        if (!fem_solver_ || !pd_solver_) return 0;

        auto& mesh = fem_solver_->mesh();
        Index num_elements = mesh.num_elements();

        // Get element damage (if available)
        // This requires FEM solver to track damage/failure
        std::vector<Real> element_damage(num_elements, 0.0);

        // TODO: Get actual damage from FEM solver
        // For now, this is a placeholder

        Index converted = 0;
        Real threshold = config_.damage_threshold;

        for (Index e = 0; e < num_elements; ++e) {
            if (element_damage[e] > threshold) {
                // Mark element for conversion
                // Add new PD particles at element nodes
                // Remove element from FEM mesh
                converted++;
            }
        }

        if (converted > 0) {
            NXS_LOG_INFO("FEMPDCoupling: Converted {} damaged elements to PD", converted);
        }

        return converted;
    }

    /**
     * @brief Get coupling statistics
     */
    void get_statistics(Index& num_fem_only, Index& num_pd_only,
                        Index& num_overlap, Index& num_coupled) const {
        num_fem_only = 0;
        num_pd_only = 0;
        num_overlap = 0;
        num_coupled = node_particle_map_.size();

        for (auto& d : fem_domain_) {
            if (d == DomainType::FEM_Only) num_fem_only++;
            else if (d == DomainType::Overlap) num_overlap++;
        }

        for (auto& d : pd_domain_) {
            if (d == DomainType::PD_Only) num_pd_only++;
        }
    }

    // Accessors
    const std::vector<NodeParticleMap>& node_particle_map() const {
        return node_particle_map_;
    }

    const std::vector<DomainType>& fem_domain() const { return fem_domain_; }
    const std::vector<DomainType>& pd_domain() const { return pd_domain_; }

private:
    /**
     * @brief Compute blending weight (Arlequin method)
     *
     * alpha = 1 at FEM boundary, 0 at PD boundary
     * Smooth transition through overlap region
     */
    Real compute_blend_weight(const Real* x, const Real* overlap_min,
                              const Real* overlap_max, Real blend_width) {
        // Distance from FEM boundary (approximate using overlap min)
        Real dist_from_fem = 0.0;
        Real overlap_size = 0.0;

        for (int d = 0; d < 3; ++d) {
            Real size = overlap_max[d] - overlap_min[d];
            if (size > 1e-10) {
                Real rel_pos = (x[d] - overlap_min[d]) / size;
                dist_from_fem = std::max(dist_from_fem, rel_pos);
                overlap_size = std::max(overlap_size, size);
            }
        }

        // Normalize by blend width
        Real t = dist_from_fem;
        if (blend_width > 0.0 && overlap_size > 0.0) {
            t = std::min(1.0, dist_from_fem * overlap_size / blend_width);
        }

        // Smooth blending function: alpha = (1 - t)^n
        Real alpha = std::pow(1.0 - t, config_.blend_exponent);
        return std::max(0.0, std::min(1.0, alpha));
    }

    FEMPDCouplingConfig config_;
    bool is_initialized_ = false;

    std::shared_ptr<FEMSolver> fem_solver_;
    std::shared_ptr<PDSolver> pd_solver_;

    std::vector<NodeParticleMap> node_particle_map_;
    std::vector<DomainType> fem_domain_;
    std::vector<DomainType> pd_domain_;
};

// ============================================================================
// Coupled FEM-PD Solver
// ============================================================================

/**
 * @brief Configuration for coupled FEM-PD simulation
 */
struct CoupledSolverConfig {
    Real dt = 1e-7;                     ///< Time step
    Index total_steps = 1000;           ///< Total steps
    Index output_interval = 100;        ///< Output interval
    Index sync_interval = 1;            ///< Coupling sync interval

    FEMPDCouplingConfig coupling;       ///< Coupling configuration
    PDSolverConfig pd_config;           ///< PD solver configuration
};

/**
 * @brief Coupled FEM-PD solver
 *
 * Runs FEM and PD solvers with interface coupling.
 * Uses staggered time stepping with synchronization.
 */
class CoupledFEMPDSolver {
public:
    CoupledFEMPDSolver() = default;

    /**
     * @brief Initialize coupled solver
     */
    void initialize(const CoupledSolverConfig& config) {
        config_ = config;

        // Initialize PD solver
        pd_solver_ = std::make_shared<PDSolver>();
        pd_solver_->initialize(config.pd_config);

        // Initialize coupling
        coupling_.initialize(config.coupling);

        NXS_LOG_INFO("CoupledFEMPDSolver initialized");
    }

    /**
     * @brief Set FEM solver
     */
    void set_fem_solver(std::shared_ptr<FEMSolver> fem_solver) {
        fem_solver_ = fem_solver;
        coupling_.set_fem_solver(fem_solver);
    }

    /**
     * @brief Set PD particles and materials
     */
    void set_pd_system(std::shared_ptr<PDParticleSystem> particles,
                       const std::vector<PDMaterial>& materials) {
        pd_solver_->set_particles(particles);
        pd_solver_->set_materials(materials);
        coupling_.set_pd_solver(pd_solver_);
    }

    /**
     * @brief Build coupling between domains
     */
    void build_coupling() {
        pd_solver_->build_neighbors();
        coupling_.build_coupling();
    }

    /**
     * @brief Perform single coupled time step
     */
    void step() {
        // 1. FEM predictor step
        if (fem_solver_) {
            fem_solver_->step(config_.dt);
        }

        // 2. Synchronize FEM to PD
        if (step_ % config_.sync_interval == 0) {
            coupling_.synchronize();
        }

        // 3. PD step
        pd_solver_->step();

        // 4. Transfer PD forces to FEM
        if (step_ % config_.sync_interval == 0) {
            coupling_.transfer_pd_to_fem();
        }

        // 5. FEM corrector step (if needed)
        // This depends on the FEM time integration scheme

        // 6. Check for damage and convert elements
        if (step_ % 10 == 0) {
            coupling_.convert_damaged_elements();
        }

        time_ += config_.dt;
        step_++;
    }

    /**
     * @brief Run coupled simulation
     */
    void run() {
        NXS_LOG_INFO("CoupledFEMPDSolver: Starting simulation");

        for (step_ = 0; step_ < config_.total_steps; ++step_) {
            step();

            if (step_ % config_.output_interval == 0) {
                Real ke_pd = pd_solver_->particles().compute_kinetic_energy();
                NXS_LOG_INFO("Step {}: time={:.2e}, PD_KE={:.2e}",
                            step_, time_, ke_pd);
            }
        }

        NXS_LOG_INFO("CoupledFEMPDSolver: Complete");
    }

    // Accessors
    PDSolver& pd_solver() { return *pd_solver_; }
    FEMPDCoupling& coupling() { return coupling_; }
    Real time() const { return time_; }
    Index step() const { return step_; }

private:
    CoupledSolverConfig config_;
    Real time_ = 0.0;
    Index step_ = 0;

    std::shared_ptr<FEMSolver> fem_solver_;
    std::shared_ptr<PDSolver> pd_solver_;
    FEMPDCoupling coupling_;
};

} // namespace pd
} // namespace nxs
