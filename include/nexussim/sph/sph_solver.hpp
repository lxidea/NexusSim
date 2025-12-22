#pragma once

/**
 * @file sph_solver.hpp
 * @brief Weakly Compressible SPH (WCSPH) solver
 *
 * Implements:
 * - Weakly compressible fluid dynamics
 * - Tait equation of state
 * - Artificial viscosity
 * - XSPH velocity correction
 * - Boundary handling
 *
 * Governing equations:
 *   dρ/dt = -ρ∇·v                  (continuity)
 *   dv/dt = -∇p/ρ + ν∇²v + g       (momentum)
 *   p = B[(ρ/ρ₀)^γ - 1]             (Tait EOS)
 *
 * References:
 * - Monaghan (1992) - Smoothed Particle Hydrodynamics
 * - Monaghan (2005) - SPH for free surface flows
 */

#include <nexussim/core/core.hpp>
#include <nexussim/sph/sph_kernel.hpp>
#include <nexussim/sph/neighbor_search.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <vector>
#include <cmath>
#include <iostream>

namespace nxs {
namespace sph {

// ============================================================================
// SPH Particle Data
// ============================================================================

/**
 * @brief SPH fluid material properties
 */
struct SPHMaterial {
    Real rho0 = 1000.0;      ///< Reference density (kg/m³)
    Real c0 = 1480.0;        ///< Speed of sound (m/s)
    Real gamma = 7.0;        ///< Tait EOS exponent
    Real mu = 0.001;         ///< Dynamic viscosity (Pa·s)
    Real surface_tension = 0.0; ///< Surface tension coefficient

    /**
     * @brief Tait equation of state: p = B[(ρ/ρ₀)^γ - 1]
     * B = ρ₀c₀²/γ
     */
    KOKKOS_INLINE_FUNCTION
    Real pressure(Real rho) const {
        Real B = rho0 * c0 * c0 / gamma;
        Real ratio = rho / rho0;
        return B * (std::pow(ratio, gamma) - 1.0);
    }

    /**
     * @brief Speed of sound for CFL
     */
    KOKKOS_INLINE_FUNCTION
    Real sound_speed(Real rho) const {
        Real ratio = rho / rho0;
        return c0 * std::pow(ratio, (gamma - 1.0) / 2.0);
    }
};

/**
 * @brief Boundary condition types
 */
enum class SPHBoundaryType {
    Free,           ///< Free surface
    Wall,           ///< Solid wall (ghost particles)
    Periodic,       ///< Periodic boundary
    Inlet,          ///< Inlet flow
    Outlet          ///< Outlet flow
};

// ============================================================================
// SPH Solver
// ============================================================================

/**
 * @brief Weakly Compressible SPH solver
 */
class SPHSolver {
public:
    SPHSolver(Real smoothing_length = 0.01)
        : h_(smoothing_length)
        , kernel_(KernelType::WendlandC2, 3)
        , grid_(smoothing_length)
        , dt_(1e-6)
        , time_(0.0)
        , cfl_(0.4)
        , alpha_visc_(0.01)     // Artificial viscosity parameter
        , beta_visc_(0.0)       // Quadratic artificial viscosity
        , xsph_epsilon_(0.5)    // XSPH velocity smoothing
        , gravity_{0.0, 0.0, -9.81}
    {}

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize particles from arrays
     */
    void initialize(const std::vector<Real>& x,
                    const std::vector<Real>& y,
                    const std::vector<Real>& z,
                    Real initial_spacing) {
        num_particles_ = x.size();
        h_ = 1.5 * initial_spacing;  // Support ratio
        grid_.set_cell_size(h_ * kernel_.support_radius());

        // Allocate arrays
        allocate(num_particles_);

        // Copy initial positions
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto rho = rho_.view_host();
        auto mass = mass_.view_host();

        Real particle_volume = initial_spacing * initial_spacing * initial_spacing;
        Real particle_mass = material_.rho0 * particle_volume;

        for (size_t i = 0; i < num_particles_; ++i) {
            pos_x(i) = x[i];
            pos_y(i) = y[i];
            pos_z(i) = z[i];
            rho(i) = material_.rho0;
            mass(i) = particle_mass;
        }

        pos_x_.modify_host(); pos_x_.sync_device();
        pos_y_.modify_host(); pos_y_.sync_device();
        pos_z_.modify_host(); pos_z_.sync_device();
        rho_.modify_host(); rho_.sync_device();
        mass_.modify_host(); mass_.sync_device();

        // Zero velocities
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();

        for (size_t i = 0; i < num_particles_; ++i) {
            vel_x(i) = 0.0;
            vel_y(i) = 0.0;
            vel_z(i) = 0.0;
        }

        vel_x_.modify_host(); vel_x_.sync_device();
        vel_y_.modify_host(); vel_y_.sync_device();
        vel_z_.modify_host(); vel_z_.sync_device();
    }

    /**
     * @brief Create dam break initial condition
     */
    void create_dam_break(Real Lx, Real Ly, Real Lz, Real spacing) {
        std::vector<Real> x, y, z;

        // Fill rectangular region with particles
        for (Real px = spacing / 2; px < Lx; px += spacing) {
            for (Real py = spacing / 2; py < Ly; py += spacing) {
                for (Real pz = spacing / 2; pz < Lz; pz += spacing) {
                    x.push_back(px);
                    y.push_back(py);
                    z.push_back(pz);
                }
            }
        }

        initialize(x, y, z, spacing);
    }

    // ========================================================================
    // Time Integration
    // ========================================================================

    /**
     * @brief Compute stable timestep from CFL
     */
    Real compute_stable_dt() const {
        // CFL: dt < CFL * h / (c + |v|_max)
        Real v_max = 0.0;

        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();

        for (size_t i = 0; i < num_particles_; ++i) {
            Real v = std::sqrt(vel_x(i) * vel_x(i) +
                              vel_y(i) * vel_y(i) +
                              vel_z(i) * vel_z(i));
            v_max = std::max(v_max, v);
        }

        Real c = material_.c0;
        return cfl_ * h_ / (c + v_max);
    }

    /**
     * @brief Advance simulation by one timestep
     */
    void step(Real dt) {
        dt_ = dt;

        // 1. Build neighbor list
        build_neighbors();

        // 2. Compute density
        compute_density();

        // 3. Compute pressure
        compute_pressure();

        // 4. Compute accelerations
        compute_acceleration();

        // 5. Integrate (Velocity-Verlet style)
        integrate(dt);

        // 6. XSPH velocity correction
        if (xsph_epsilon_ > 0) {
            apply_xsph_correction();
        }

        // 7. Apply boundary conditions
        apply_boundary_conditions();

        time_ += dt;
    }

    // ========================================================================
    // SPH Computations
    // ========================================================================

    void build_neighbors() {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();

        // Build spatial hash
        grid_.build(&pos_x(0), &pos_y(0), &pos_z(0), num_particles_);

        // Find all neighbor pairs
        Real support = h_ * kernel_.support_radius();
        grid_.find_neighbors(&pos_x(0), &pos_y(0), &pos_z(0), support, neighbor_pairs_);

        // Build compact neighbor list
        neighbor_list_.build_from_pairs(neighbor_pairs_, num_particles_);
    }

    void compute_density() {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto rho = rho_.view_host();
        auto mass = mass_.view_host();

        // Initialize with self contribution
        for (size_t i = 0; i < num_particles_; ++i) {
            rho(i) = mass(i) * kernel_.W(0, h_);
        }

        // Sum neighbor contributions
        for (const auto& pair : neighbor_pairs_) {
            Real W = kernel_.W(pair.r, h_);

            rho(pair.i) += mass(pair.j) * W;
            rho(pair.j) += mass(pair.i) * W;
        }

        rho_.modify_host();
    }

    void compute_pressure() {
        auto rho = rho_.view_host();
        auto p = pressure_.view_host();

        for (size_t i = 0; i < num_particles_; ++i) {
            p(i) = material_.pressure(rho(i));
        }

        pressure_.modify_host();
    }

    void compute_acceleration() {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();
        auto rho = rho_.view_host();
        auto p = pressure_.view_host();
        auto mass = mass_.view_host();
        auto acc_x = acc_x_.view_host();
        auto acc_y = acc_y_.view_host();
        auto acc_z = acc_z_.view_host();

        // Initialize with gravity
        for (size_t i = 0; i < num_particles_; ++i) {
            acc_x(i) = gravity_[0];
            acc_y(i) = gravity_[1];
            acc_z(i) = gravity_[2];
        }

        // Pressure and viscosity contributions
        for (const auto& pair : neighbor_pairs_) {
            Index i = pair.i;
            Index j = pair.j;

            // Kernel gradient
            Real gWx, gWy, gWz;
            kernel_.grad_W_vec(pair.rx, pair.ry, pair.rz, h_, gWx, gWy, gWz);

            // Pressure term: -∑ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
            Real p_term = p(i) / (rho(i) * rho(i)) + p(j) / (rho(j) * rho(j));

            acc_x(i) -= mass(j) * p_term * gWx;
            acc_y(i) -= mass(j) * p_term * gWy;
            acc_z(i) -= mass(j) * p_term * gWz;

            acc_x(j) += mass(i) * p_term * gWx;
            acc_y(j) += mass(i) * p_term * gWy;
            acc_z(j) += mass(i) * p_term * gWz;

            // Artificial viscosity (Monaghan)
            Real vx_ij = vel_x(i) - vel_x(j);
            Real vy_ij = vel_y(i) - vel_y(j);
            Real vz_ij = vel_z(i) - vel_z(j);

            Real v_dot_r = vx_ij * pair.rx + vy_ij * pair.ry + vz_ij * pair.rz;

            if (v_dot_r < 0) {  // Approaching particles
                Real c_avg = 0.5 * (material_.sound_speed(rho(i)) +
                                   material_.sound_speed(rho(j)));
                Real rho_avg = 0.5 * (rho(i) + rho(j));

                Real mu = h_ * v_dot_r / (pair.r * pair.r + 0.01 * h_ * h_);
                Real Pi_ij = (-alpha_visc_ * c_avg * mu +
                              beta_visc_ * mu * mu) / rho_avg;

                acc_x(i) -= mass(j) * Pi_ij * gWx;
                acc_y(i) -= mass(j) * Pi_ij * gWy;
                acc_z(i) -= mass(j) * Pi_ij * gWz;

                acc_x(j) += mass(i) * Pi_ij * gWx;
                acc_y(j) += mass(i) * Pi_ij * gWy;
                acc_z(j) += mass(i) * Pi_ij * gWz;
            }
        }

        acc_x_.modify_host();
        acc_y_.modify_host();
        acc_z_.modify_host();
    }

    void integrate(Real dt) {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();
        auto acc_x = acc_x_.view_host();
        auto acc_y = acc_y_.view_host();
        auto acc_z = acc_z_.view_host();

        // Velocity-Verlet (symplectic)
        for (size_t i = 0; i < num_particles_; ++i) {
            // v^{n+1/2} = v^n + (dt/2) * a^n
            vel_x(i) += 0.5 * dt * acc_x(i);
            vel_y(i) += 0.5 * dt * acc_y(i);
            vel_z(i) += 0.5 * dt * acc_z(i);

            // x^{n+1} = x^n + dt * v^{n+1/2}
            pos_x(i) += dt * vel_x(i);
            pos_y(i) += dt * vel_y(i);
            pos_z(i) += dt * vel_z(i);

            // a^{n+1} will be computed next iteration
            // v^{n+1} = v^{n+1/2} + (dt/2) * a^{n+1} (deferred)
        }

        pos_x_.modify_host();
        pos_y_.modify_host();
        pos_z_.modify_host();
        vel_x_.modify_host();
        vel_y_.modify_host();
        vel_z_.modify_host();
    }

    void apply_xsph_correction() {
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();
        auto rho = rho_.view_host();
        auto mass = mass_.view_host();

        // XSPH: v_corrected = v + ε * Σ m_j/ρ_avg * (v_j - v_i) * W
        std::vector<Real> dvx(num_particles_, 0.0);
        std::vector<Real> dvy(num_particles_, 0.0);
        std::vector<Real> dvz(num_particles_, 0.0);

        for (const auto& pair : neighbor_pairs_) {
            Index i = pair.i;
            Index j = pair.j;

            Real W = kernel_.W(pair.r, h_);
            Real rho_avg = 0.5 * (rho(i) + rho(j));

            Real factor_i = xsph_epsilon_ * mass(j) / rho_avg * W;
            Real factor_j = xsph_epsilon_ * mass(i) / rho_avg * W;

            dvx[i] += factor_i * (vel_x(j) - vel_x(i));
            dvy[i] += factor_i * (vel_y(j) - vel_y(i));
            dvz[i] += factor_i * (vel_z(j) - vel_z(i));

            dvx[j] += factor_j * (vel_x(i) - vel_x(j));
            dvy[j] += factor_j * (vel_y(i) - vel_y(j));
            dvz[j] += factor_j * (vel_z(i) - vel_z(j));
        }

        for (size_t i = 0; i < num_particles_; ++i) {
            vel_x(i) += dvx[i];
            vel_y(i) += dvy[i];
            vel_z(i) += dvz[i];
        }

        vel_x_.modify_host();
        vel_y_.modify_host();
        vel_z_.modify_host();
    }

    void apply_boundary_conditions() {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();

        // Simple reflecting boundary (floor at z=0)
        for (size_t i = 0; i < num_particles_; ++i) {
            if (pos_z(i) < h_) {
                pos_z(i) = h_;
                vel_z(i) = std::abs(vel_z(i)) * 0.5;  // Damped reflection
            }

            // Side walls
            if (pos_x(i) < 0) { pos_x(i) = 0; vel_x(i) = std::abs(vel_x(i)) * 0.5; }
            if (pos_y(i) < 0) { pos_y(i) = 0; vel_y(i) = std::abs(vel_y(i)) * 0.5; }

            if (pos_x(i) > domain_size_[0]) {
                pos_x(i) = domain_size_[0];
                vel_x(i) = -std::abs(vel_x(i)) * 0.5;
            }
            if (pos_y(i) > domain_size_[1]) {
                pos_y(i) = domain_size_[1];
                vel_y(i) = -std::abs(vel_y(i)) * 0.5;
            }
        }

        pos_x_.modify_host();
        pos_y_.modify_host();
        pos_z_.modify_host();
        vel_x_.modify_host();
        vel_y_.modify_host();
        vel_z_.modify_host();
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    void set_material(const SPHMaterial& mat) { material_ = mat; }
    void set_smoothing_length(Real h) { h_ = h; grid_.set_cell_size(h * kernel_.support_radius()); }
    void set_kernel(KernelType type) { kernel_ = SPHKernel(type, 3); }
    void set_gravity(Real gx, Real gy, Real gz) { gravity_ = {gx, gy, gz}; }
    void set_cfl(Real cfl) { cfl_ = cfl; }
    void set_artificial_viscosity(Real alpha, Real beta = 0.0) { alpha_visc_ = alpha; beta_visc_ = beta; }
    void set_xsph_epsilon(Real eps) { xsph_epsilon_ = eps; }
    void set_domain_size(Real Lx, Real Ly, Real Lz) { domain_size_ = {Lx, Ly, Lz}; }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t num_particles() const { return num_particles_; }
    Real time() const { return time_; }
    Real smoothing_length() const { return h_; }
    const SPHMaterial& material() const { return material_; }

    // Get particle data (read-only)
    auto positions_x() const { return pos_x_.view_host(); }
    auto positions_y() const { return pos_y_.view_host(); }
    auto positions_z() const { return pos_z_.view_host(); }
    auto velocities_x() const { return vel_x_.view_host(); }
    auto velocities_y() const { return vel_y_.view_host(); }
    auto velocities_z() const { return vel_z_.view_host(); }
    auto densities() const { return rho_.view_host(); }
    auto pressures() const { return pressure_.view_host(); }

    /**
     * @brief Compute total kinetic energy
     */
    Real kinetic_energy() const {
        auto vel_x = vel_x_.view_host();
        auto vel_y = vel_y_.view_host();
        auto vel_z = vel_z_.view_host();
        auto mass = mass_.view_host();

        Real KE = 0.0;
        for (size_t i = 0; i < num_particles_; ++i) {
            Real v2 = vel_x(i) * vel_x(i) + vel_y(i) * vel_y(i) + vel_z(i) * vel_z(i);
            KE += 0.5 * mass(i) * v2;
        }
        return KE;
    }

    /**
     * @brief Compute center of mass
     */
    void center_of_mass(Real& cx, Real& cy, Real& cz) const {
        auto pos_x = pos_x_.view_host();
        auto pos_y = pos_y_.view_host();
        auto pos_z = pos_z_.view_host();
        auto mass = mass_.view_host();

        Real total_mass = 0.0;
        cx = cy = cz = 0.0;

        for (size_t i = 0; i < num_particles_; ++i) {
            cx += mass(i) * pos_x(i);
            cy += mass(i) * pos_y(i);
            cz += mass(i) * pos_z(i);
            total_mass += mass(i);
        }

        if (total_mass > 0) {
            cx /= total_mass;
            cy /= total_mass;
            cz /= total_mass;
        }
    }

    void print_stats(std::ostream& os = std::cout) const {
        auto rho = rho_.view_host();
        auto p = pressure_.view_host();

        Real rho_min = rho(0), rho_max = rho(0);
        Real p_min = p(0), p_max = p(0);

        for (size_t i = 0; i < num_particles_; ++i) {
            rho_min = std::min(rho_min, rho(i));
            rho_max = std::max(rho_max, rho(i));
            p_min = std::min(p_min, p(i));
            p_max = std::max(p_max, p(i));
        }

        os << "=== SPH Solver Statistics ===\n";
        os << "Particles: " << num_particles_ << "\n";
        os << "Time: " << time_ << " s\n";
        os << "Smoothing length: " << h_ << " m\n";
        os << "Density range: [" << rho_min << ", " << rho_max << "] kg/m³\n";
        os << "Pressure range: [" << p_min << ", " << p_max << "] Pa\n";
        os << "Kinetic energy: " << kinetic_energy() << " J\n";
        os << "Avg neighbors: " << neighbor_list_.avg_neighbors() << "\n";
        os << "=============================\n";
    }

private:
    void allocate(size_t n) {
        pos_x_ = Kokkos::DualView<Real*>("pos_x", n);
        pos_y_ = Kokkos::DualView<Real*>("pos_y", n);
        pos_z_ = Kokkos::DualView<Real*>("pos_z", n);
        vel_x_ = Kokkos::DualView<Real*>("vel_x", n);
        vel_y_ = Kokkos::DualView<Real*>("vel_y", n);
        vel_z_ = Kokkos::DualView<Real*>("vel_z", n);
        acc_x_ = Kokkos::DualView<Real*>("acc_x", n);
        acc_y_ = Kokkos::DualView<Real*>("acc_y", n);
        acc_z_ = Kokkos::DualView<Real*>("acc_z", n);
        rho_ = Kokkos::DualView<Real*>("rho", n);
        pressure_ = Kokkos::DualView<Real*>("pressure", n);
        mass_ = Kokkos::DualView<Real*>("mass", n);
    }

    // Solver parameters
    Real h_;                ///< Smoothing length
    SPHKernel kernel_;      ///< Kernel function
    SpatialHashGrid grid_;  ///< Neighbor search
    Real dt_;               ///< Current timestep
    Real time_;             ///< Current time
    Real cfl_;              ///< CFL number
    Real alpha_visc_;       ///< Artificial viscosity (linear)
    Real beta_visc_;        ///< Artificial viscosity (quadratic)
    Real xsph_epsilon_;     ///< XSPH smoothing parameter
    std::array<Real, 3> gravity_;    ///< Gravity vector
    std::array<Real, 3> domain_size_ = {1.0, 1.0, 1.0}; ///< Domain size

    // Material
    SPHMaterial material_;

    // Particle data
    size_t num_particles_ = 0;
    Kokkos::DualView<Real*> pos_x_, pos_y_, pos_z_;
    Kokkos::DualView<Real*> vel_x_, vel_y_, vel_z_;
    Kokkos::DualView<Real*> acc_x_, acc_y_, acc_z_;
    Kokkos::DualView<Real*> rho_;
    Kokkos::DualView<Real*> pressure_;
    Kokkos::DualView<Real*> mass_;

    // Neighbor data
    std::vector<NeighborPair> neighbor_pairs_;
    CompactNeighborList neighbor_list_;
};

} // namespace sph
} // namespace nxs
