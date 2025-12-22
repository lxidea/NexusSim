/**
 * @file fem_solver.cpp
 * @brief FEM solver implementation
 */

#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/discretization/tet10.hpp>
#include <nexussim/discretization/wedge6.hpp>
#include <nexussim/discretization/shell4.hpp>
#include <nexussim/discretization/beam2.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/core/gpu.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

FEMSolver::FEMSolver(const std::string& name)
    : PhysicsModule(name, PhysicsModule::Type::FEM)
    , num_nodes_(0)
    , ndof_(0)
    , dof_per_node_(3)  // Default to 3 DOFs (3D solid)
    , cfl_factor_(0.9)
    , damping_factor_(0.0)
    , gravity_enabled_(false)
    , body_force_enabled_(false)
{
    // Initialize gravity and body force
    gravity_[0] = 0.0;
    gravity_[1] = 0.0;
    gravity_[2] = -9.81;  // Default gravity in -z direction
    body_force_[0] = 0.0;
    body_force_[1] = 0.0;
    body_force_[2] = 0.0;

    // Create default explicit integrator
    integrator_ = std::make_shared<physics::ExplicitCentralDifferenceIntegrator>();
}

// ============================================================================
// Initialization
// ============================================================================

void FEMSolver::initialize(std::shared_ptr<Mesh> mesh,
                           std::shared_ptr<State> state)
{
    NXS_LOG_INFO("Initializing FEM solver '{}'", name_);

    // Initialize Kokkos if not already initialized
    if (!KokkosManager::instance().is_initialized()) {
        NXS_LOG_INFO("Initializing Kokkos for GPU acceleration");
        KokkosManager::instance().initialize();
    }

    mesh_ = mesh;
    state_ = state;

    // Get number of nodes from mesh
    const auto& coords = mesh_->coordinates();
    num_nodes_ = coords.num_entities();
    ndof_ = num_nodes_ * dof_per_node_;

    NXS_LOG_INFO("  Nodes: {}, DOFs: {}", num_nodes_, ndof_);

    // Allocate state vectors as Kokkos::DualView (host + device)
    displacement_ = Kokkos::DualView<Real*>("displacement", ndof_);
    velocity_ = Kokkos::DualView<Real*>("velocity", ndof_);
    acceleration_ = Kokkos::DualView<Real*>("acceleration", ndof_);
    force_internal_ = Kokkos::DualView<Real*>("force_internal", ndof_);
    force_external_ = Kokkos::DualView<Real*>("force_external", ndof_);
    mass_ = Kokkos::DualView<Real*>("mass", ndof_);

    // Initialize to zero on host
    auto disp_h = displacement_.view_host();
    auto vel_h = velocity_.view_host();
    auto acc_h = acceleration_.view_host();
    auto f_int_h = force_internal_.view_host();
    auto f_ext_h = force_external_.view_host();
    auto mass_h = mass_.view_host();

    for (std::size_t i = 0; i < ndof_; ++i) {
        disp_h(i) = 0.0;
        vel_h(i) = 0.0;
        acc_h(i) = 0.0;
        f_int_h(i) = 0.0;
        f_ext_h(i) = 0.0;
        mass_h(i) = 0.0;
    }

    // Mark host data as modified (for lazy synchronization)
    displacement_.modify_host();
    velocity_.modify_host();
    acceleration_.modify_host();
    force_internal_.modify_host();
    force_external_.modify_host();
    mass_.modify_host();

    // Cache node coordinates on device (one-time copy for performance)
    NXS_LOG_INFO("Caching node coordinates on device");
    coords_device_ = Kokkos::View<Real**>("coords_device", num_nodes_, 3);
    auto coords_h = Kokkos::create_mirror_view(coords_device_);

    for (std::size_t n = 0; n < num_nodes_; ++n) {
        coords_h(n, 0) = coords.at(n, 0);
        coords_h(n, 1) = coords.at(n, 1);
        coords_h(n, 2) = coords.at(n, 2);
    }

    Kokkos::deep_copy(coords_device_, coords_h);
    coords_device_ready_ = true;
    NXS_LOG_INFO("  Coordinates cached: {} nodes", num_nodes_);

    // Initialize time integrator
    integrator_->initialize(ndof_);

    // Assemble mass matrix
    assemble_mass_matrix();

    // No longer need explicit zero-mass checking here since it's handled
    // in assemble_mass_matrix() with proper redistribution

    set_status(Status::Ready);
    NXS_LOG_INFO("FEM solver initialized successfully");
}

void FEMSolver::finalize()
{
    NXS_LOG_INFO("Finalizing FEM solver '{}'", name_);
    PhysicsModule::finalize();
}

// ============================================================================
// Time Integration
// ============================================================================

void FEMSolver::step(Real dt)
{
    // Zero out internal forces
    zero_forces();

    // Apply force boundary conditions (external loads)
    apply_force_boundary_conditions(current_time_);

    // Apply gravity and body forces
    apply_body_forces();

    // Compute internal forces from current displacement
    compute_internal_forces();

    // ========================================================================
    // GPU-ACCELERATED TIME INTEGRATION (Phase 2A)
    // ========================================================================

    // Sync data to device for GPU computation
    // Note: force_internal_ is already on device from compute_internal_forces() - DON'T sync it!
    // Note: force_external_ was modified on host by apply_force_boundary_conditions()
    displacement_.sync_device();
    velocity_.sync_device();
    acceleration_.sync_device();
    force_external_.sync_device();
    // force_internal_.sync_device();  // DON'T sync - already computed on device!
    mass_.sync_device();

    // Get device views
    auto disp_d = displacement_.view_device();
    auto vel_d = velocity_.view_device();
    auto acc_d = acceleration_.view_device();
    auto f_ext_d = force_external_.view_device();
    auto f_int_d = force_internal_.view_device();
    auto mass_d = mass_.view_device();

    // Capture damping factor for lambda
    const Real damp = damping_factor_;

    // GPU Parallel Time Integration using Explicit Central Difference
    // a^n = M^{-1} * (f_ext^n - f_int^n)
    // v^{n+1} = v^n + a^n * dt - damping * v^n
    // u^{n+1} = u^n + v^{n+1} * dt
    Kokkos::parallel_for("TimeIntegration", ndof_, KOKKOS_LAMBDA(const int i) {
        // Compute acceleration: a = (f_ext - f_int) / m
        const Real net_force = f_ext_d(i) - f_int_d(i);
        acc_d(i) = net_force / mass_d(i);

        // Update velocity with damping: v += a*dt
        // Rayleigh mass damping: f_damp = -α*M*v, so a_damp = -α*v
        // v^{n+1} = v^n + a*dt - α*v^n*dt = v^n*(1-α*dt) + a*dt
        vel_d(i) += acc_d(i) * dt;
        if (damp > 0.0) {
            // damp is Rayleigh α coefficient (angular frequency units)
            // Scale by dt to get per-step damping factor, clamped to [0,1]
            const Real damp_factor = (damp * dt < 1.0) ? damp * dt : 0.99;
            vel_d(i) *= (1.0 - damp_factor);
        }

        // Update displacement: u += v*dt
        disp_d(i) += vel_d(i) * dt;
    });

    // Wait for GPU to finish
    Kokkos::fence("TimeIntegration_complete");

    // Mark device data as modified
    displacement_.modify_device();
    velocity_.modify_device();
    acceleration_.modify_device();

    // Sync to host for boundary conditions (they're applied on host for now)
    displacement_.sync_host();
    velocity_.sync_host();
    acceleration_.sync_host();

    // Apply essential (displacement) boundary conditions after integration
    apply_displacement_boundary_conditions(current_time_);

    // Constrain zero-mass DOFs (unused nodes) to prevent spurious motion
    if (!zero_mass_dofs_.empty()) {
        auto disp_h = displacement_.view_host();
        auto vel_h = velocity_.view_host();
        auto acc_h = acceleration_.view_host();

        for (const Index dof : zero_mass_dofs_) {
            disp_h(dof) = 0.0;
            vel_h(dof) = 0.0;
            acc_h(dof) = 0.0;
        }

        displacement_.modify_host();
        velocity_.modify_host();
        acceleration_.modify_host();
    }

    // Advance time
    advance_time(dt);
}

Real FEMSolver::compute_stable_dt() const
{
    if (element_groups_.empty()) {
        NXS_LOG_WARN("No element groups defined, using default dt = 1.0");
        return 1.0;
    }

    Real min_dt = std::numeric_limits<Real>::max();

    // Loop over element groups
    for (const auto& group : element_groups_) {
        const Real wave_speed = compute_wave_speed(group.material);

        // Loop over elements in group
        const std::size_t num_elems = group.element_ids.size();
        for (std::size_t e = 0; e < num_elems; ++e) {
            const Real elem_size = compute_element_size(group, group.element_ids[e]);

            // CFL condition: dt < L / c
            const Real dt_elem = cfl_factor_ * elem_size / wave_speed;
            min_dt = std::min(min_dt, dt_elem);
        }
    }

    return min_dt;
}

// ============================================================================
// Assembly
// ============================================================================

void FEMSolver::assemble_mass_matrix()
{
    NXS_LOG_INFO("Assembling global mass matrix");

    // Zero mass matrix (on host)
    auto mass_h = mass_.view_host();
    for (std::size_t i = 0; i < ndof_; ++i) {
        mass_h(i) = 0.0;
    }

    // Get node coordinates
    const auto& coords = mesh_->coordinates();

    // Loop over element groups
    for (const auto& group : element_groups_) {
        const std::size_t num_elems = group.element_ids.size();
        const auto& elem = group.element;
        const auto props = elem->properties();
        const std::size_t nodes_per_elem = props.num_nodes;
        const std::size_t dof_per_elem = nodes_per_elem * props.num_dof_per_node;

        // Element mass matrix
        std::vector<Real> Me(dof_per_elem * dof_per_elem);
        std::vector<Real> elem_coords(nodes_per_elem * 3);

        // Loop over elements
        for (std::size_t e = 0; e < num_elems; ++e) {
            // Get element connectivity
            const std::size_t conn_offset = e * nodes_per_elem;

            // Extract element coordinates
            for (std::size_t n = 0; n < nodes_per_elem; ++n) {
                const Index node = group.connectivity[conn_offset + n];
                elem_coords[n * 3 + 0] = coords.at(node, 0);
                elem_coords[n * 3 + 1] = coords.at(node, 1);
                elem_coords[n * 3 + 2] = coords.at(node, 2);
            }

            // Lump mass matrix
            // Use HRZ (Hinton-Rock-Zienkiewicz) lumping for quadratic elements (Hex20, Tet10)
            // to avoid negative masses, otherwise use standard row-sum lumping
            std::vector<Real> Me_lumped(dof_per_elem, 0.0);

            if (props.type == physics::ElementType::Hex20) {
                // HRZ lumping for Hex20: compute nodal masses directly via integration
                // Create a temporary Hex20Element to call the lumping method
                Hex20Element hex20_elem;
                std::vector<Real> nodal_mass(nodes_per_elem, 0.0);
                hex20_elem.lumped_mass_hrz(elem_coords.data(), group.material.density, nodal_mass.data());

                // Replicate nodal mass across 3 DOFs (x, y, z) for each node
                for (std::size_t n = 0; n < nodes_per_elem; ++n) {
                    for (Index d = 0; d < props.num_dof_per_node; ++d) {
                        Me_lumped[n * props.num_dof_per_node + d] = nodal_mass[n];
                    }
                }
            } else {
                // Standard row-sum lumping for other elements
                // First compute consistent mass matrix
                std::vector<Real> Me(dof_per_elem * dof_per_elem, 0.0);
                elem->mass_matrix(elem_coords.data(), group.material.density, Me.data());

                // Row-sum to get lumped mass
                for (std::size_t i = 0; i < dof_per_elem; ++i) {
                    for (std::size_t j = 0; j < dof_per_elem; ++j) {
                        Me_lumped[i] += Me[i * dof_per_elem + j];
                    }
                }

                // For quadratic elements (Tet10, etc.), row-sum lumping can still produce negative
                // masses. Redistribute negative masses proportionally to positive masses
                Real total_lumped = 0.0;
                Real negative_sum = 0.0;
                for (std::size_t i = 0; i < dof_per_elem; ++i) {
                    total_lumped += Me_lumped[i];
                    if (Me_lumped[i] < 0.0) {
                        negative_sum += Me_lumped[i];
                    }
                }

                // If we have negative masses, redistribute them
                if (negative_sum < -1.0e-10) {  // Tolerance for numerical errors
                    Real positive_sum = total_lumped - negative_sum;  // Sum of positive masses
                    for (std::size_t i = 0; i < dof_per_elem; ++i) {
                        if (Me_lumped[i] < 0.0) {
                            Me_lumped[i] = 0.0;  // Zero out negative masses
                        } else if (positive_sum > 1.0e-10) {
                            // Scale up positive masses to conserve total mass
                            Me_lumped[i] *= (total_lumped / positive_sum);
                        }
                    }
                }
            }

            // For 3D elements: each node's mass is replicated across 3 DOFs (x,y,z)
            // We need to divide by NUM_DIMS so that the total mass equals physical mass
            const Real dof_scaling = (props.num_dof_per_node == 3) ? 1.0 / 3.0 : 1.0;

            // Assemble into global mass
            for (std::size_t n = 0; n < nodes_per_elem; ++n) {
                const Index node = group.connectivity[conn_offset + n];
                for (Index d = 0; d < props.num_dof_per_node; ++d) {
                    const Index global_dof = node * dof_per_node_ + d;
                    const Index local_dof = n * props.num_dof_per_node + d;
                    mass_h(global_dof) += Me_lumped[local_dof] * dof_scaling;
                }
            }
        }
    }

    // Global redistribution of zero/negative masses
    // This is needed for quadratic elements where row-sum lumping can produce
    // zero/negative masses at some nodes even after per-element redistribution
    // Also handles unused nodes (e.g., in structured meshes where not all grid nodes are used)
    Real global_total_mass = 0.0;
    Real positive_sum = 0.0;
    std::size_t num_zero_or_negative = 0;

    for (std::size_t i = 0; i < ndof_; ++i) {
        global_total_mass += mass_h(i);
        if (mass_h(i) > 1.0e-10) {
            positive_sum += mass_h(i);
        } else {
            num_zero_or_negative++;
        }
    }

    // If there are zero/negative masses, fix them to avoid division by zero
    // Store indices of zero-mass DOFs for later constraint application
    std::vector<Index> zero_mass_dofs;
    if (num_zero_or_negative > 0) {
        NXS_LOG_WARN("Found {} zero/negative mass DOFs out of {} total DOFs",
                     num_zero_or_negative, ndof_);

        if (positive_sum > 1.0e-10) {
            // Compute appropriate small mass based on the actual element masses
            // Use average POSITIVE mass to ensure numerical stability
            const Real avg_positive_mass = positive_sum / (ndof_ - num_zero_or_negative);
            const Real small_mass = avg_positive_mass;  // Same as average to blend in

            std::size_t fixed_count = 0;
            for (std::size_t i = 0; i < ndof_; ++i) {
                if (mass_h(i) <= 1.0e-10) {
                    // Set to small mass to avoid division by zero
                    // These DOFs will be constrained to zero displacement below
                    mass_h(i) = small_mass;
                    zero_mass_dofs.push_back(i);
                    fixed_count++;
                }
            }

            NXS_LOG_WARN("  Assigned mass ({:.3e} kg) to {} DOFs",
                         small_mass, fixed_count);
            NXS_LOG_WARN("  These DOFs (unused nodes) will be constrained to prevent spurious motion");
        } else {
            // No positive masses at all - this is a critical error
            NXS_LOG_ERROR("ERROR: All mass DOFs are zero or negative! Cannot initialize solver.");
            throw std::runtime_error("Mass matrix has no positive entries");
        }
    }

    // Store zero-mass DOFs for constraint application during time integration
    zero_mass_dofs_ = zero_mass_dofs;

    // Mark mass as modified on host
    mass_.modify_host();

    Real total_mass = 0.0;
    for (std::size_t i = 0; i < ndof_; ++i) {
        total_mass += mass_h(i);
    }
    NXS_LOG_INFO("  Total mass: {:.6e} kg", total_mass);
}

// Helper function to prepare GPU data for element groups
void FEMSolver::prepare_gpu_element_data(ElementGroup& group)
{
    if (group.gpu_data_ready) return;

    const std::size_t conn_size = group.connectivity.size();

    // Allocate device memory for connectivity
    group.connectivity_device = Kokkos::View<Index*>("connectivity", conn_size);

    // Copy connectivity to device
    auto conn_h = Kokkos::create_mirror_view(group.connectivity_device);
    for (std::size_t i = 0; i < conn_size; ++i) {
        conn_h(i) = group.connectivity[i];
    }
    Kokkos::deep_copy(group.connectivity_device, conn_h);

    group.gpu_data_ready = true;
}

void FEMSolver::compute_internal_forces()
{
    // Sync displacement to device
    displacement_.sync_device();

    // Sync force_internal to device (zero_forces modified it on host)
    force_internal_.sync_device();

    auto disp_d = displacement_.view_device();
    auto f_int_d = force_internal_.view_device();

    // Zero forces on device (redundant but ensures it's zero)
    Kokkos::parallel_for("ZeroInternalForces", ndof_, KOKKOS_LAMBDA(const int i) {
        f_int_d(i) = 0.0;
    });

    // Use cached device coordinates (optimization: avoid repeated host-device transfers)
    auto coords_d = coords_device_;

    // Keep reference to mesh coordinates for CPU fallback paths
    const auto& coords = mesh_->coordinates();

    bool used_gpu_kernel = false;

    // Loop over element groups
    for (auto& group : element_groups_) {
        const std::size_t num_elems = group.element_ids.size();
        const auto& elem = group.element;
        const auto props = elem->properties();
        const std::size_t nodes_per_elem = props.num_nodes;
        const std::size_t dof_per_elem = nodes_per_elem * props.num_dof_per_node;

        // Prepare GPU data for this group
        prepare_gpu_element_data(group);

        // Material constants
        const Real E = group.material.E;
        const Real nu = group.material.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        // Get connectivity device view
        auto conn_d = group.connectivity_device;
        const Index dof_per_node = dof_per_node_;

        // GPU kernel for Hex8 elements
        if (props.type == physics::ElementType::Hex8) {
            NXS_LOG_INFO("Using GPU kernel for Hex8 elements ({} elements)", num_elems);

            Kokkos::parallel_for("Hex8_ElementForces", num_elems, KOKKOS_LAMBDA(const int e) {
                // Stack arrays for element data (no dynamic allocation on GPU!)
                const int nnodes = 8;
                const int ndof_elem = 24; // 8 nodes × 3 DOFs

                Real elem_coords[24]; // 8 nodes × 3 coords
                Real elem_disp[24];
                Real elem_force[24];

                // Initialize
                for (int i = 0; i < ndof_elem; ++i) {
                    elem_force[i] = 0.0;
                }

                // Extract element coordinates and displacements
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);

                    // Coordinates
                    elem_coords[n * 3 + 0] = coords_d(node, 0);
                    elem_coords[n * 3 + 1] = coords_d(node, 1);
                    elem_coords[n * 3 + 2] = coords_d(node, 2);

                    // Displacements
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        elem_disp[n * 3 + d] = disp_d(global_dof);
                    }
                }

                // Gauss integration (2×2×2 for Hex8)
                const Real gp = 0.577350269189626;  // 1/sqrt(3)
                const Real gp_coords[8][3] = {
                    {-gp, -gp, -gp}, { gp, -gp, -gp},
                    { gp,  gp, -gp}, {-gp,  gp, -gp},
                    {-gp, -gp,  gp}, { gp, -gp,  gp},
                    { gp,  gp,  gp}, {-gp,  gp,  gp}
                };

                // Loop over Gauss points
                for (int gp_idx = 0; gp_idx < 8; ++gp_idx) {
                    const Real xi = gp_coords[gp_idx][0];
                    const Real eta = gp_coords[gp_idx][1];
                    const Real zeta = gp_coords[gp_idx][2];

                    // Compute shape function derivatives (B-matrix)
                    Real dN_dxi[24]; // 8 nodes × 3 derivatives

                    // dN/dxi for each node
                    dN_dxi[0*3+0] = -0.125 * (1-eta) * (1-zeta);
                    dN_dxi[1*3+0] =  0.125 * (1-eta) * (1-zeta);
                    dN_dxi[2*3+0] =  0.125 * (1+eta) * (1-zeta);
                    dN_dxi[3*3+0] = -0.125 * (1+eta) * (1-zeta);
                    dN_dxi[4*3+0] = -0.125 * (1-eta) * (1+zeta);
                    dN_dxi[5*3+0] =  0.125 * (1-eta) * (1+zeta);
                    dN_dxi[6*3+0] =  0.125 * (1+eta) * (1+zeta);
                    dN_dxi[7*3+0] = -0.125 * (1+eta) * (1+zeta);

                    // dN/deta
                    dN_dxi[0*3+1] = -0.125 * (1-xi) * (1-zeta);
                    dN_dxi[1*3+1] = -0.125 * (1+xi) * (1-zeta);
                    dN_dxi[2*3+1] =  0.125 * (1+xi) * (1-zeta);
                    dN_dxi[3*3+1] =  0.125 * (1-xi) * (1-zeta);
                    dN_dxi[4*3+1] = -0.125 * (1-xi) * (1+zeta);
                    dN_dxi[5*3+1] = -0.125 * (1+xi) * (1+zeta);
                    dN_dxi[6*3+1] =  0.125 * (1+xi) * (1+zeta);
                    dN_dxi[7*3+1] =  0.125 * (1-xi) * (1+zeta);

                    // dN/dzeta
                    dN_dxi[0*3+2] = -0.125 * (1-xi) * (1-eta);
                    dN_dxi[1*3+2] = -0.125 * (1+xi) * (1-eta);
                    dN_dxi[2*3+2] = -0.125 * (1+xi) * (1+eta);
                    dN_dxi[3*3+2] = -0.125 * (1-xi) * (1+eta);
                    dN_dxi[4*3+2] =  0.125 * (1-xi) * (1-eta);
                    dN_dxi[5*3+2] =  0.125 * (1+xi) * (1-eta);
                    dN_dxi[6*3+2] =  0.125 * (1+xi) * (1+eta);
                    dN_dxi[7*3+2] =  0.125 * (1-xi) * (1+eta);

                    // Compute Jacobian: J = dX/dxi
                    Real J[9] = {0};
                    for (int n = 0; n < nnodes; ++n) {
                        const Real x = elem_coords[n*3+0];
                        const Real y = elem_coords[n*3+1];
                        const Real z = elem_coords[n*3+2];

                        J[0] += dN_dxi[n*3+0] * x;  // dx/dxi
                        J[1] += dN_dxi[n*3+0] * y;  // dy/dxi
                        J[2] += dN_dxi[n*3+0] * z;  // dz/dxi
                        J[3] += dN_dxi[n*3+1] * x;  // dx/deta
                        J[4] += dN_dxi[n*3+1] * y;  // dy/deta
                        J[5] += dN_dxi[n*3+1] * z;  // dz/deta
                        J[6] += dN_dxi[n*3+2] * x;  // dx/dzeta
                        J[7] += dN_dxi[n*3+2] * y;  // dy/dzeta
                        J[8] += dN_dxi[n*3+2] * z;  // dz/dzeta
                    }

                    // Determinant of Jacobian
                    const Real detJ = J[0]*(J[4]*J[8]-J[5]*J[7])
                                    - J[1]*(J[3]*J[8]-J[5]*J[6])
                                    + J[2]*(J[3]*J[7]-J[4]*J[6]);

                    // Inverse Jacobian
                    Real Jinv[9];
                    const Real invDetJ = 1.0 / detJ;
                    Jinv[0] = (J[4]*J[8] - J[5]*J[7]) * invDetJ;
                    Jinv[1] = (J[2]*J[7] - J[1]*J[8]) * invDetJ;
                    Jinv[2] = (J[1]*J[5] - J[2]*J[4]) * invDetJ;
                    Jinv[3] = (J[5]*J[6] - J[3]*J[8]) * invDetJ;
                    Jinv[4] = (J[0]*J[8] - J[2]*J[6]) * invDetJ;
                    Jinv[5] = (J[2]*J[3] - J[0]*J[5]) * invDetJ;
                    Jinv[6] = (J[3]*J[7] - J[4]*J[6]) * invDetJ;
                    Jinv[7] = (J[1]*J[6] - J[0]*J[7]) * invDetJ;
                    Jinv[8] = (J[0]*J[4] - J[1]*J[3]) * invDetJ;

                    // Compute dN/dx = Jinv * dN/dxi
                    Real dN_dx[24] = {0};
                    for (int n = 0; n < nnodes; ++n) {
                        dN_dx[n*3+0] = Jinv[0]*dN_dxi[n*3+0] + Jinv[1]*dN_dxi[n*3+1] + Jinv[2]*dN_dxi[n*3+2];
                        dN_dx[n*3+1] = Jinv[3]*dN_dxi[n*3+0] + Jinv[4]*dN_dxi[n*3+1] + Jinv[5]*dN_dxi[n*3+2];
                        dN_dx[n*3+2] = Jinv[6]*dN_dxi[n*3+0] + Jinv[7]*dN_dxi[n*3+1] + Jinv[8]*dN_dxi[n*3+2];
                    }

                    // Compute strain: ε = B * u
                    Real strain[6] = {0};  // εxx, εyy, εzz, γxy, γyz, γxz
                    for (int n = 0; n < nnodes; ++n) {
                        const Real ux = elem_disp[n*3+0];
                        const Real uy = elem_disp[n*3+1];
                        const Real uz = elem_disp[n*3+2];

                        strain[0] += dN_dx[n*3+0] * ux;  // εxx = du/dx
                        strain[1] += dN_dx[n*3+1] * uy;  // εyy = dv/dy
                        strain[2] += dN_dx[n*3+2] * uz;  // εzz = dw/dz
                        strain[3] += dN_dx[n*3+1] * ux + dN_dx[n*3+0] * uy;  // γxy
                        strain[4] += dN_dx[n*3+2] * uy + dN_dx[n*3+1] * uz;  // γyz
                        strain[5] += dN_dx[n*3+0] * uz + dN_dx[n*3+2] * ux;  // γxz
                    }

                    // Compute stress: σ = C * ε (isotropic linear elastic)
                    Real stress[6];
                    stress[0] = (lambda + 2.0*mu)*strain[0] + lambda*(strain[1] + strain[2]);
                    stress[1] = (lambda + 2.0*mu)*strain[1] + lambda*(strain[0] + strain[2]);
                    stress[2] = (lambda + 2.0*mu)*strain[2] + lambda*(strain[0] + strain[1]);
                    stress[3] = mu * strain[3];
                    stress[4] = mu * strain[4];
                    stress[5] = mu * strain[5];

                    // Accumulate internal forces: f = B^T * σ * detJ * weight
                    const Real weight = 1.0;  // Weight for 2×2×2 Gauss quadrature
                    const Real factor = detJ * weight;

                    for (int n = 0; n < nnodes; ++n) {
                        elem_force[n*3+0] += (dN_dx[n*3+0]*stress[0] + dN_dx[n*3+1]*stress[3] + dN_dx[n*3+2]*stress[5]) * factor;
                        elem_force[n*3+1] += (dN_dx[n*3+0]*stress[3] + dN_dx[n*3+1]*stress[1] + dN_dx[n*3+2]*stress[4]) * factor;
                        elem_force[n*3+2] += (dN_dx[n*3+0]*stress[5] + dN_dx[n*3+1]*stress[4] + dN_dx[n*3+2]*stress[2]) * factor;
                    }
                }

                // Assemble to global force vector using atomic operations
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        Kokkos::atomic_add(&f_int_d(global_dof), elem_force[n*3+d]);
                    }
                }
            });

            Kokkos::fence("Hex8_ElementForces_complete");
            used_gpu_kernel = true;

            continue;  // Skip CPU fallback for Hex8
        }

        // GPU kernel for Tet4 elements
        if (props.type == physics::ElementType::Tet4) {
            NXS_LOG_INFO("Using GPU kernel for Tet4 elements ({} elements)", num_elems);

            Kokkos::parallel_for("Tet4_ElementForces", num_elems, KOKKOS_LAMBDA(const int e) {
                const int nnodes = 4;
                const int ndof_elem = 12; // 4 nodes × 3 DOFs

                Real elem_coords[12]; // 4 nodes × 3 coords
                Real elem_disp[12];
                Real elem_force[12];

                // Initialize
                for (int i = 0; i < ndof_elem; ++i) {
                    elem_force[i] = 0.0;
                }

                // Extract element coordinates and displacements
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);

                    elem_coords[n * 3 + 0] = coords_d(node, 0);
                    elem_coords[n * 3 + 1] = coords_d(node, 1);
                    elem_coords[n * 3 + 2] = coords_d(node, 2);

                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        elem_disp[n * 3 + d] = disp_d(global_dof);
                    }
                }

                // For Tet4, shape function derivatives are constant
                // N1 = 1 - xi - eta - zeta
                // N2 = xi
                // N3 = eta
                // N4 = zeta

                // Compute Jacobian: J = [x2-x1, x3-x1, x4-x1]
                //                       [y2-y1, y3-y1, y4-y1]
                //                       [z2-z1, z3-z1, z4-z1]
                Real J[9];
                J[0] = elem_coords[3] - elem_coords[0];  // x2 - x1
                J[1] = elem_coords[6] - elem_coords[0];  // x3 - x1
                J[2] = elem_coords[9] - elem_coords[0];  // x4 - x1
                J[3] = elem_coords[4] - elem_coords[1];  // y2 - y1
                J[4] = elem_coords[7] - elem_coords[1];  // y3 - y1
                J[5] = elem_coords[10] - elem_coords[1]; // y4 - y1
                J[6] = elem_coords[5] - elem_coords[2];  // z2 - z1
                J[7] = elem_coords[8] - elem_coords[2];  // z3 - z1
                J[8] = elem_coords[11] - elem_coords[2]; // z4 - z1

                // Determinant
                const Real detJ = J[0]*(J[4]*J[8] - J[5]*J[7])
                                - J[1]*(J[3]*J[8] - J[5]*J[6])
                                + J[2]*(J[3]*J[7] - J[4]*J[6]);

                // Volume = detJ / 6
                const Real volume = detJ / 6.0;

                // Inverse Jacobian
                Real Jinv[9];
                const Real invDetJ = 1.0 / detJ;
                Jinv[0] = (J[4]*J[8] - J[5]*J[7]) * invDetJ;
                Jinv[1] = (J[2]*J[7] - J[1]*J[8]) * invDetJ;
                Jinv[2] = (J[1]*J[5] - J[2]*J[4]) * invDetJ;
                Jinv[3] = (J[5]*J[6] - J[3]*J[8]) * invDetJ;
                Jinv[4] = (J[0]*J[8] - J[2]*J[6]) * invDetJ;
                Jinv[5] = (J[2]*J[3] - J[0]*J[5]) * invDetJ;
                Jinv[6] = (J[3]*J[7] - J[4]*J[6]) * invDetJ;
                Jinv[7] = (J[1]*J[6] - J[0]*J[7]) * invDetJ;
                Jinv[8] = (J[0]*J[4] - J[1]*J[3]) * invDetJ;

                // Shape function derivatives in natural coordinates (constant for Tet4)
                // dN/dxi = [-1, 1, 0, 0]
                // dN/deta = [-1, 0, 1, 0]
                // dN/dzeta = [-1, 0, 0, 1]

                // Compute dN/dx = Jinv^T * dN/dxi
                Real dN_dx[12]; // 4 nodes × 3 derivatives

                // Node 1: dN1/dxi = -1, dN1/deta = -1, dN1/dzeta = -1
                dN_dx[0*3+0] = -Jinv[0] - Jinv[1] - Jinv[2];
                dN_dx[0*3+1] = -Jinv[3] - Jinv[4] - Jinv[5];
                dN_dx[0*3+2] = -Jinv[6] - Jinv[7] - Jinv[8];

                // Node 2: dN2/dxi = 1, dN2/deta = 0, dN2/dzeta = 0
                dN_dx[1*3+0] = Jinv[0];
                dN_dx[1*3+1] = Jinv[3];
                dN_dx[1*3+2] = Jinv[6];

                // Node 3: dN3/dxi = 0, dN3/deta = 1, dN3/dzeta = 0
                dN_dx[2*3+0] = Jinv[1];
                dN_dx[2*3+1] = Jinv[4];
                dN_dx[2*3+2] = Jinv[7];

                // Node 4: dN4/dxi = 0, dN4/deta = 0, dN4/dzeta = 1
                dN_dx[3*3+0] = Jinv[2];
                dN_dx[3*3+1] = Jinv[5];
                dN_dx[3*3+2] = Jinv[8];

                // Compute strain
                Real strain[6] = {0};
                for (int n = 0; n < nnodes; ++n) {
                    const Real ux = elem_disp[n*3+0];
                    const Real uy = elem_disp[n*3+1];
                    const Real uz = elem_disp[n*3+2];

                    strain[0] += dN_dx[n*3+0] * ux;  // εxx
                    strain[1] += dN_dx[n*3+1] * uy;  // εyy
                    strain[2] += dN_dx[n*3+2] * uz;  // εzz
                    strain[3] += dN_dx[n*3+1] * ux + dN_dx[n*3+0] * uy;  // γxy
                    strain[4] += dN_dx[n*3+2] * uy + dN_dx[n*3+1] * uz;  // γyz
                    strain[5] += dN_dx[n*3+0] * uz + dN_dx[n*3+2] * ux;  // γxz
                }

                // Compute stress
                Real stress[6];
                stress[0] = (lambda + 2.0*mu)*strain[0] + lambda*(strain[1] + strain[2]);
                stress[1] = (lambda + 2.0*mu)*strain[1] + lambda*(strain[0] + strain[2]);
                stress[2] = (lambda + 2.0*mu)*strain[2] + lambda*(strain[0] + strain[1]);
                stress[3] = mu * strain[3];
                stress[4] = mu * strain[4];
                stress[5] = mu * strain[5];

                // Compute forces: f = B^T * σ * volume
                for (int n = 0; n < nnodes; ++n) {
                    elem_force[n*3+0] = (dN_dx[n*3+0]*stress[0] + dN_dx[n*3+1]*stress[3] + dN_dx[n*3+2]*stress[5]) * volume;
                    elem_force[n*3+1] = (dN_dx[n*3+0]*stress[3] + dN_dx[n*3+1]*stress[1] + dN_dx[n*3+2]*stress[4]) * volume;
                    elem_force[n*3+2] = (dN_dx[n*3+0]*stress[5] + dN_dx[n*3+1]*stress[4] + dN_dx[n*3+2]*stress[2]) * volume;
                }

                // Assemble to global
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        Kokkos::atomic_add(&f_int_d(global_dof), elem_force[n*3+d]);
                    }
                }
            });

            Kokkos::fence("Tet4_ElementForces_complete");
            used_gpu_kernel = true;

            continue;  // Skip CPU fallback for Tet4
        }

        // ====================================================================
        // Hex20 GPU Kernel (20-node hexahedron, quadratic shape functions)
        // ====================================================================
        if (false && props.type == physics::ElementType::Hex20) {  // DISABLE GPU - test CPU fallback
            NXS_LOG_INFO("Using GPU kernel for Hex20 elements ({} elements)", num_elems);

            // Prepare GPU data
            prepare_gpu_element_data(group);

            // Get device views
            force_internal_.sync_device();
            displacement_.sync_device();

            auto f_int_d = force_internal_.view_device();
            auto disp_d = displacement_.view_device();
            auto conn_d = group.connectivity_device;

            // Material constants
            const Real E = group.material.E;
            const Real nu = group.material.nu;
            const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            const Real mu = E / (2.0 * (1.0 + nu));

            // 3x3x3 Gauss quadrature for quadratic elements (27 points)
            const Real sqrt_3_5 = 0.7745966692414834;  // √(3/5)
            const Real gp_coords[3] = {-sqrt_3_5, 0.0, sqrt_3_5};
            const Real gp_weights[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

            // Get DOF per node for proper indexing
            const Index dof_per_node = dof_per_node_;

            // Launch GPU kernel
            Kokkos::parallel_for("Hex20_ElementForces", num_elems, KOKKOS_LAMBDA(const int e) {
                const int nnodes = 20;
                const int ndof_elem = 60;

                // Stack arrays (no dynamic allocation on GPU)
                Real elem_coords[60];  // 20 nodes × 3 coords
                Real elem_disp[60];    // 20 nodes × 3 DOFs
                Real elem_force[60];   // Accumulated internal forces

                // Initialize forces to zero
                for (int i = 0; i < ndof_elem; ++i) {
                    elem_force[i] = 0.0;
                }

                // Extract element data from global arrays
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    elem_coords[n*3 + 0] = coords_d(node, 0);
                    elem_coords[n*3 + 1] = coords_d(node, 1);
                    elem_coords[n*3 + 2] = coords_d(node, 2);

                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        elem_disp[n*3 + d] = disp_d(global_dof);
                    }
                }

                // 27-point Gauss integration (3x3x3)
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            const Real xi = gp_coords[i];
                            const Real eta = gp_coords[j];
                            const Real zeta = gp_coords[k];
                            const Real w = gp_weights[i] * gp_weights[j] * gp_weights[k];

                            // ========================================
                            // Hex20 Serendipity Shape Derivatives
                            // ========================================
                            Real dN[60];  // 20 nodes × 3 derivatives

                            const Real xi2 = xi * xi;
                            const Real eta2 = eta * eta;
                            const Real zeta2 = zeta * zeta;

                            // Corner nodes (0-7): Complex quadratic shape functions
                            // dN/dξ
                            dN[0*3 + 0] = -0.125 * (1.0 - eta) * (1.0 - zeta) * (-2.0*xi - eta - zeta - 1.0);
                            dN[1*3 + 0] = 0.125 * (1.0 - eta) * (1.0 - zeta) * (2.0*xi - eta - zeta - 1.0);
                            dN[2*3 + 0] = 0.125 * (1.0 + eta) * (1.0 - zeta) * (2.0*xi + eta - zeta - 1.0);
                            dN[3*3 + 0] = -0.125 * (1.0 + eta) * (1.0 - zeta) * (-2.0*xi + eta - zeta - 1.0);
                            dN[4*3 + 0] = -0.125 * (1.0 - eta) * (1.0 + zeta) * (-2.0*xi - eta + zeta - 1.0);
                            dN[5*3 + 0] = 0.125 * (1.0 - eta) * (1.0 + zeta) * (2.0*xi - eta + zeta - 1.0);
                            dN[6*3 + 0] = 0.125 * (1.0 + eta) * (1.0 + zeta) * (2.0*xi + eta + zeta - 1.0);
                            dN[7*3 + 0] = -0.125 * (1.0 + eta) * (1.0 + zeta) * (-2.0*xi + eta + zeta - 1.0);

                            // dN/dη
                            dN[0*3 + 1] = -0.125 * (1.0 - xi) * (1.0 - zeta) * (-xi - 2.0*eta - zeta - 1.0);
                            dN[1*3 + 1] = -0.125 * (1.0 + xi) * (1.0 - zeta) * (xi - 2.0*eta - zeta - 1.0);
                            dN[2*3 + 1] = 0.125 * (1.0 + xi) * (1.0 - zeta) * (xi + 2.0*eta - zeta - 1.0);
                            dN[3*3 + 1] = 0.125 * (1.0 - xi) * (1.0 - zeta) * (-xi + 2.0*eta - zeta - 1.0);
                            dN[4*3 + 1] = -0.125 * (1.0 - xi) * (1.0 + zeta) * (-xi - 2.0*eta + zeta - 1.0);
                            dN[5*3 + 1] = -0.125 * (1.0 + xi) * (1.0 + zeta) * (xi - 2.0*eta + zeta - 1.0);
                            dN[6*3 + 1] = 0.125 * (1.0 + xi) * (1.0 + zeta) * (xi + 2.0*eta + zeta - 1.0);
                            dN[7*3 + 1] = 0.125 * (1.0 - xi) * (1.0 + zeta) * (-xi + 2.0*eta + zeta - 1.0);

                            // dN/dζ
                            dN[0*3 + 2] = -0.125 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 2.0*zeta - 1.0);
                            dN[1*3 + 2] = -0.125 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 2.0*zeta - 1.0);
                            dN[2*3 + 2] = -0.125 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 2.0*zeta - 1.0);
                            dN[3*3 + 2] = -0.125 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 2.0*zeta - 1.0);
                            dN[4*3 + 2] = 0.125 * (1.0 - xi) * (1.0 - eta) * (-xi - eta + 2.0*zeta - 1.0);
                            dN[5*3 + 2] = 0.125 * (1.0 + xi) * (1.0 - eta) * (xi - eta + 2.0*zeta - 1.0);
                            dN[6*3 + 2] = 0.125 * (1.0 + xi) * (1.0 + eta) * (xi + eta + 2.0*zeta - 1.0);
                            dN[7*3 + 2] = 0.125 * (1.0 - xi) * (1.0 + eta) * (-xi + eta + 2.0*zeta - 1.0);

                            // Mid-edge nodes (8-19): Simpler derivatives
                            // Bottom face edges (zeta=-1)
                            dN[8*3 + 0] = -0.5 * xi * (1.0 - eta) * (1.0 - zeta);
                            dN[8*3 + 1] = -0.25 * (1.0 - xi2) * (1.0 - zeta);
                            dN[8*3 + 2] = -0.25 * (1.0 - xi2) * (1.0 - eta);

                            dN[9*3 + 0] = 0.25 * (1.0 - eta2) * (1.0 - zeta);
                            dN[9*3 + 1] = -0.5 * eta * (1.0 + xi) * (1.0 - zeta);
                            dN[9*3 + 2] = -0.25 * (1.0 + xi) * (1.0 - eta2);

                            dN[10*3 + 0] = -0.5 * xi * (1.0 + eta) * (1.0 - zeta);
                            dN[10*3 + 1] = 0.25 * (1.0 - xi2) * (1.0 - zeta);
                            dN[10*3 + 2] = -0.25 * (1.0 - xi2) * (1.0 + eta);

                            dN[11*3 + 0] = -0.25 * (1.0 - eta2) * (1.0 - zeta);
                            dN[11*3 + 1] = -0.5 * eta * (1.0 - xi) * (1.0 - zeta);
                            dN[11*3 + 2] = -0.25 * (1.0 - xi) * (1.0 - eta2);

                            // Vertical edges
                            dN[12*3 + 0] = -0.25 * (1.0 - eta) * (1.0 - zeta2);
                            dN[12*3 + 1] = -0.25 * (1.0 - xi) * (1.0 - zeta2);
                            dN[12*3 + 2] = -0.5 * zeta * (1.0 - xi) * (1.0 - eta);

                            dN[13*3 + 0] = 0.25 * (1.0 - eta) * (1.0 - zeta2);
                            dN[13*3 + 1] = -0.25 * (1.0 + xi) * (1.0 - zeta2);
                            dN[13*3 + 2] = -0.5 * zeta * (1.0 + xi) * (1.0 - eta);

                            dN[14*3 + 0] = 0.25 * (1.0 + eta) * (1.0 - zeta2);
                            dN[14*3 + 1] = 0.25 * (1.0 + xi) * (1.0 - zeta2);
                            dN[14*3 + 2] = -0.5 * zeta * (1.0 + xi) * (1.0 + eta);

                            dN[15*3 + 0] = -0.25 * (1.0 + eta) * (1.0 - zeta2);
                            dN[15*3 + 1] = 0.25 * (1.0 - xi) * (1.0 - zeta2);
                            dN[15*3 + 2] = -0.5 * zeta * (1.0 - xi) * (1.0 + eta);

                            // Top face edges (zeta=+1)
                            dN[16*3 + 0] = -0.5 * xi * (1.0 - eta) * (1.0 + zeta);
                            dN[16*3 + 1] = -0.25 * (1.0 - xi2) * (1.0 + zeta);
                            dN[16*3 + 2] = 0.25 * (1.0 - xi2) * (1.0 - eta);

                            dN[17*3 + 0] = 0.25 * (1.0 - eta2) * (1.0 + zeta);
                            dN[17*3 + 1] = -0.5 * eta * (1.0 + xi) * (1.0 + zeta);
                            dN[17*3 + 2] = 0.25 * (1.0 + xi) * (1.0 - eta2);

                            dN[18*3 + 0] = -0.5 * xi * (1.0 + eta) * (1.0 + zeta);
                            dN[18*3 + 1] = 0.25 * (1.0 - xi2) * (1.0 + zeta);
                            dN[18*3 + 2] = 0.25 * (1.0 - xi2) * (1.0 + eta);

                            dN[19*3 + 0] = -0.25 * (1.0 - eta2) * (1.0 + zeta);
                            dN[19*3 + 1] = -0.5 * eta * (1.0 - xi) * (1.0 + zeta);
                            dN[19*3 + 2] = 0.25 * (1.0 - xi) * (1.0 - eta2);

                            // ========================================
                            // Jacobian Matrix: J = dX/dξ (column-major to match hex20.cpp)
                            // ========================================
                            Real J[9] = {0};  // 3x3 matrix
                            for (int n = 0; n < nnodes; ++n) {
                                const Real x = elem_coords[n*3 + 0];
                                const Real y = elem_coords[n*3 + 1];
                                const Real z = elem_coords[n*3 + 2];

                                // Column-major: First column is ∂/∂ξ
                                J[0] += dN[n*3 + 0] * x;  // ∂x/∂ξ
                                J[3] += dN[n*3 + 0] * y;  // ∂y/∂ξ
                                J[6] += dN[n*3 + 0] * z;  // ∂z/∂ξ

                                // Second column is ∂/∂η
                                J[1] += dN[n*3 + 1] * x;  // ∂x/∂η
                                J[4] += dN[n*3 + 1] * y;  // ∂y/∂η
                                J[7] += dN[n*3 + 1] * z;  // ∂z/∂η

                                // Third column is ∂/∂ζ
                                J[2] += dN[n*3 + 2] * x;  // ∂x/∂ζ
                                J[5] += dN[n*3 + 2] * y;  // ∂y/∂ζ
                                J[8] += dN[n*3 + 2] * z;  // ∂z/∂ζ
                            }

                            // Determinant
                            const Real detJ = J[0]*(J[4]*J[8] - J[5]*J[7])
                                            - J[1]*(J[3]*J[8] - J[5]*J[6])
                                            + J[2]*(J[3]*J[7] - J[4]*J[6]);

                            // Debug: Check for negative or very small Jacobian
                            if (detJ <= 1.0e-15) {
                                // Skip degenerate element - this Gauss point contributes no force
                                continue;
                            }

                            // Inverse Jacobian
                            const Real invDetJ = 1.0 / detJ;
                            Real J_inv[9];
                            J_inv[0] = (J[4]*J[8] - J[5]*J[7]) * invDetJ;
                            J_inv[1] = (J[2]*J[7] - J[1]*J[8]) * invDetJ;
                            J_inv[2] = (J[1]*J[5] - J[2]*J[4]) * invDetJ;
                            J_inv[3] = (J[5]*J[6] - J[3]*J[8]) * invDetJ;
                            J_inv[4] = (J[0]*J[8] - J[2]*J[6]) * invDetJ;
                            J_inv[5] = (J[2]*J[3] - J[0]*J[5]) * invDetJ;
                            J_inv[6] = (J[3]*J[7] - J[4]*J[6]) * invDetJ;
                            J_inv[7] = (J[1]*J[6] - J[0]*J[7]) * invDetJ;
                            J_inv[8] = (J[0]*J[4] - J[1]*J[3]) * invDetJ;

                            // ========================================
                            // Global derivatives: dN/dx = J^{-1} * dN/dξ
                            // TESTING: Negate to fix sign error
                            // ========================================
                            Real dN_dx[60];  // 20 nodes × 3 global derivatives
                            for (int n = 0; n < nnodes; ++n) {
                                dN_dx[n*3 + 0] = -(J_inv[0]*dN[n*3+0] + J_inv[1]*dN[n*3+1] + J_inv[2]*dN[n*3+2]);  // NEGATED
                                dN_dx[n*3 + 1] = -(J_inv[3]*dN[n*3+0] + J_inv[4]*dN[n*3+1] + J_inv[5]*dN[n*3+2]);  // NEGATED
                                dN_dx[n*3 + 2] = -(J_inv[6]*dN[n*3+0] + J_inv[7]*dN[n*3+1] + J_inv[8]*dN[n*3+2]);  // NEGATED
                            }

                            // ========================================
                            // Strain (Voigt notation)
                            // ========================================
                            Real strain[6] = {0};
                            for (int n = 0; n < nnodes; ++n) {
                                const Real ux = elem_disp[n*3 + 0];
                                const Real uy = elem_disp[n*3 + 1];
                                const Real uz = elem_disp[n*3 + 2];

                                strain[0] += dN_dx[n*3 + 0] * ux;  // εxx
                                strain[1] += dN_dx[n*3 + 1] * uy;  // εyy
                                strain[2] += dN_dx[n*3 + 2] * uz;  // εzz
                                strain[3] += dN_dx[n*3 + 1] * ux + dN_dx[n*3 + 0] * uy;  // γxy
                                strain[4] += dN_dx[n*3 + 2] * uy + dN_dx[n*3 + 1] * uz;  // γyz
                                strain[5] += dN_dx[n*3 + 2] * ux + dN_dx[n*3 + 0] * uz;  // γxz
                            }

                            // ========================================
                            // Stress (linear elastic)
                            // ========================================
                            Real stress[6];
                            const Real trace = strain[0] + strain[1] + strain[2];
                            stress[0] = lambda * trace + 2.0 * mu * strain[0];
                            stress[1] = lambda * trace + 2.0 * mu * strain[1];
                            stress[2] = lambda * trace + 2.0 * mu * strain[2];
                            stress[3] = mu * strain[3];
                            stress[4] = mu * strain[4];
                            stress[5] = mu * strain[5];

                            // ========================================
                            // Internal forces: f = B^T * σ * detJ * w
                            // ========================================
                            const Real factor = detJ * w;
                            for (int n = 0; n < nnodes; ++n) {
                                elem_force[n*3 + 0] += factor * (dN_dx[n*3+0]*stress[0] +
                                                                  dN_dx[n*3+1]*stress[3] +
                                                                  dN_dx[n*3+2]*stress[5]);
                                elem_force[n*3 + 1] += factor * (dN_dx[n*3+1]*stress[1] +
                                                                  dN_dx[n*3+0]*stress[3] +
                                                                  dN_dx[n*3+2]*stress[4]);
                                elem_force[n*3 + 2] += factor * (dN_dx[n*3+2]*stress[2] +
                                                                  dN_dx[n*3+1]*stress[4] +
                                                                  dN_dx[n*3+0]*stress[5]);
                            }
                        }  // k loop (zeta)
                    }  // j loop (eta)
                }  // i loop (xi)

                // ========================================
                // Assemble to global (atomic for thread safety)
                // ========================================
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * dof_per_node + d;
                        Kokkos::atomic_add(&f_int_d(global_dof), elem_force[n*3 + d]);
                    }
                }
            });

            Kokkos::fence("Hex20_ElementForces_complete");
            used_gpu_kernel = true;

            continue;  // Skip CPU fallback for Hex20
        }

        // ====================================================================
        // Tet10 GPU Kernel (10-node tetrahedron, quadratic shape functions)
        // ====================================================================
        if (props.type == physics::ElementType::Tet10) {
            NXS_LOG_INFO("Using GPU kernel for Tet10 elements ({} elements)", num_elems);

            // Prepare GPU data
            prepare_gpu_element_data(group);

            // Get device views
            force_internal_.sync_device();
            displacement_.sync_device();

            auto f_int_d = force_internal_.view_device();
            auto disp_d = displacement_.view_device();
            auto conn_d = group.connectivity_device;

            // Material constants
            const Real E = group.material.E;
            const Real nu = group.material.nu;
            const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            const Real mu = E / (2.0 * (1.0 + nu));

            // 4-point Gauss quadrature for quadratic tet
            const Real a = 0.585410196624969;  // (5 + 3*sqrt(5))/20
            const Real b = 0.138196601125011;  // (5 - sqrt(5))/20
            const Real gp_coords[4][3] = {{a, b, b}, {b, a, b}, {b, b, a}, {b, b, b}};
            const Real gp_weights[4] = {0.25, 0.25, 0.25, 0.25};

            // Launch GPU kernel
            Kokkos::parallel_for("Tet10_ElementForces", num_elems, KOKKOS_LAMBDA(const int e) {
                const int nnodes = 10;
                const int ndof_elem = 30;

                // Stack arrays
                Real elem_coords[30];  // 10 nodes × 3 coords
                Real elem_disp[30];
                Real elem_force[30];

                // Initialize forces to zero
                for (int i = 0; i < ndof_elem; ++i) {
                    elem_force[i] = 0.0;
                }

                // Extract element data
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    elem_coords[n*3 + 0] = coords_d(node, 0);
                    elem_coords[n*3 + 1] = coords_d(node, 1);
                    elem_coords[n*3 + 2] = coords_d(node, 2);

                    elem_disp[n*3 + 0] = disp_d(node * 3 + 0);
                    elem_disp[n*3 + 1] = disp_d(node * 3 + 1);
                    elem_disp[n*3 + 2] = disp_d(node * 3 + 2);
                }

                // 4-point Gauss integration
                for (int gp_idx = 0; gp_idx < 4; ++gp_idx) {
                    const Real L1 = gp_coords[gp_idx][0];
                    const Real L2 = gp_coords[gp_idx][1];
                    const Real L3 = gp_coords[gp_idx][2];
                    const Real L4 = 1.0 - L1 - L2 - L3;
                    const Real w = gp_weights[gp_idx];

                    // ========================================
                    // Tet10 Shape Derivatives (barycentric coords)
                    // ========================================
                    Real dN[30];  // 10 nodes × 3 derivatives

                    // Corner nodes: dN_i/dL = 4*L_i - 1
                    dN[0*3 + 0] = 4.0 * L1 - 1.0;
                    dN[0*3 + 1] = 0.0;
                    dN[0*3 + 2] = 0.0;

                    dN[1*3 + 0] = 0.0;
                    dN[1*3 + 1] = 4.0 * L2 - 1.0;
                    dN[1*3 + 2] = 0.0;

                    dN[2*3 + 0] = 0.0;
                    dN[2*3 + 1] = 0.0;
                    dN[2*3 + 2] = 4.0 * L3 - 1.0;

                    dN[3*3 + 0] = -(4.0 * L4 - 1.0);
                    dN[3*3 + 1] = -(4.0 * L4 - 1.0);
                    dN[3*3 + 2] = -(4.0 * L4 - 1.0);

                    // Edge nodes: N = 4*L_j*L_k
                    dN[4*3 + 0] = 4.0 * L2;  // Edge 0-1
                    dN[4*3 + 1] = 4.0 * L1;
                    dN[4*3 + 2] = 0.0;

                    dN[5*3 + 0] = 0.0;  // Edge 1-2
                    dN[5*3 + 1] = 4.0 * L3;
                    dN[5*3 + 2] = 4.0 * L2;

                    dN[6*3 + 0] = 4.0 * L3;  // Edge 2-0
                    dN[6*3 + 1] = 0.0;
                    dN[6*3 + 2] = 4.0 * L1;

                    dN[7*3 + 0] = 4.0 * (L4 - L1);  // Edge 0-3
                    dN[7*3 + 1] = -4.0 * L1;
                    dN[7*3 + 2] = -4.0 * L1;

                    dN[8*3 + 0] = -4.0 * L2;  // Edge 1-3
                    dN[8*3 + 1] = 4.0 * (L4 - L2);
                    dN[8*3 + 2] = -4.0 * L2;

                    dN[9*3 + 0] = -4.0 * L3;  // Edge 2-3
                    dN[9*3 + 1] = -4.0 * L3;
                    dN[9*3 + 2] = 4.0 * (L4 - L3);

                    // ========================================
                    // Jacobian Matrix
                    // ========================================
                    Real J[9] = {0};
                    for (int n = 0; n < nnodes; ++n) {
                        const Real x = elem_coords[n*3 + 0];
                        const Real y = elem_coords[n*3 + 1];
                        const Real z = elem_coords[n*3 + 2];

                        J[0] += dN[n*3 + 0] * x;  // dX/dL1
                        J[1] += dN[n*3 + 1] * x;  // dX/dL2
                        J[2] += dN[n*3 + 2] * x;  // dX/dL3

                        J[3] += dN[n*3 + 0] * y;
                        J[4] += dN[n*3 + 1] * y;
                        J[5] += dN[n*3 + 2] * y;

                        J[6] += dN[n*3 + 0] * z;
                        J[7] += dN[n*3 + 1] * z;
                        J[8] += dN[n*3 + 2] * z;
                    }

                    // Determinant
                    const Real detJ = J[0]*(J[4]*J[8] - J[5]*J[7])
                                    - J[1]*(J[3]*J[8] - J[5]*J[6])
                                    + J[2]*(J[3]*J[7] - J[4]*J[6]);

                    if (detJ <= 0.0) continue;  // Skip degenerate element

                    // Inverse Jacobian
                    const Real invDetJ = 1.0 / detJ;
                    Real J_inv[9];
                    J_inv[0] = (J[4]*J[8] - J[5]*J[7]) * invDetJ;
                    J_inv[1] = (J[2]*J[7] - J[1]*J[8]) * invDetJ;
                    J_inv[2] = (J[1]*J[5] - J[2]*J[4]) * invDetJ;
                    J_inv[3] = (J[5]*J[6] - J[3]*J[8]) * invDetJ;
                    J_inv[4] = (J[0]*J[8] - J[2]*J[6]) * invDetJ;
                    J_inv[5] = (J[2]*J[3] - J[0]*J[5]) * invDetJ;
                    J_inv[6] = (J[3]*J[7] - J[4]*J[6]) * invDetJ;
                    J_inv[7] = (J[1]*J[6] - J[0]*J[7]) * invDetJ;
                    J_inv[8] = (J[0]*J[4] - J[1]*J[3]) * invDetJ;

                    // ========================================
                    // Global derivatives
                    // ========================================
                    Real dN_dx[30];
                    for (int n = 0; n < nnodes; ++n) {
                        dN_dx[n*3 + 0] = J_inv[0]*dN[n*3+0] + J_inv[1]*dN[n*3+1] + J_inv[2]*dN[n*3+2];
                        dN_dx[n*3 + 1] = J_inv[3]*dN[n*3+0] + J_inv[4]*dN[n*3+1] + J_inv[5]*dN[n*3+2];
                        dN_dx[n*3 + 2] = J_inv[6]*dN[n*3+0] + J_inv[7]*dN[n*3+1] + J_inv[8]*dN[n*3+2];
                    }

                    // ========================================
                    // Strain (Voigt notation)
                    // ========================================
                    Real strain[6] = {0};
                    for (int n = 0; n < nnodes; ++n) {
                        const Real ux = elem_disp[n*3 + 0];
                        const Real uy = elem_disp[n*3 + 1];
                        const Real uz = elem_disp[n*3 + 2];

                        strain[0] += dN_dx[n*3 + 0] * ux;  // εxx
                        strain[1] += dN_dx[n*3 + 1] * uy;  // εyy
                        strain[2] += dN_dx[n*3 + 2] * uz;  // εzz
                        strain[3] += dN_dx[n*3 + 1] * ux + dN_dx[n*3 + 0] * uy;  // γxy
                        strain[4] += dN_dx[n*3 + 2] * uy + dN_dx[n*3 + 1] * uz;  // γyz
                        strain[5] += dN_dx[n*3 + 2] * ux + dN_dx[n*3 + 0] * uz;  // γxz
                    }

                    // ========================================
                    // Stress (linear elastic)
                    // ========================================
                    Real stress[6];
                    const Real trace = strain[0] + strain[1] + strain[2];
                    stress[0] = lambda * trace + 2.0 * mu * strain[0];
                    stress[1] = lambda * trace + 2.0 * mu * strain[1];
                    stress[2] = lambda * trace + 2.0 * mu * strain[2];
                    stress[3] = mu * strain[3];
                    stress[4] = mu * strain[4];
                    stress[5] = mu * strain[5];

                    // ========================================
                    // Internal forces: f = B^T * σ * detJ * w
                    // ========================================
                    const Real factor = detJ * w;
                    for (int n = 0; n < nnodes; ++n) {
                        elem_force[n*3 + 0] += factor * (dN_dx[n*3+0]*stress[0] +
                                                          dN_dx[n*3+1]*stress[3] +
                                                          dN_dx[n*3+2]*stress[5]);
                        elem_force[n*3 + 1] += factor * (dN_dx[n*3+1]*stress[1] +
                                                          dN_dx[n*3+0]*stress[3] +
                                                          dN_dx[n*3+2]*stress[4]);
                        elem_force[n*3 + 2] += factor * (dN_dx[n*3+2]*stress[2] +
                                                          dN_dx[n*3+1]*stress[4] +
                                                          dN_dx[n*3+0]*stress[5]);
                    }
                }  // Gauss point loop

                // ========================================
                // Assemble to global (atomic for thread safety)
                // ========================================
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * 3 + d;
                        Kokkos::atomic_add(&f_int_d(global_dof), elem_force[n*3 + d]);
                    }
                }
            });

            Kokkos::fence("Tet10_ElementForces_complete");
            used_gpu_kernel = true;

            continue;  // Skip CPU fallback for Tet10
        }

        // ====================================================================
        // Wedge6 GPU Kernel (6-node prism/wedge element)
        // ====================================================================
        if (props.type == physics::ElementType::Wedge6) {
            NXS_LOG_INFO("Using GPU kernel for Wedge6 elements ({} elements)", num_elems);

            // Prepare GPU data
            prepare_gpu_element_data(group);

            // Get device views
            force_internal_.sync_device();
            displacement_.sync_device();

            auto f_int_d = force_internal_.view_device();
            auto disp_d = displacement_.view_device();
            auto conn_d = group.connectivity_device;

            // Material constants
            const Real E = group.material.E;
            const Real nu = group.material.nu;
            const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            const Real mu = E / (2.0 * (1.0 + nu));

            // 6-point Gauss quadrature: 3-point triangle × 2-point ζ
            const Real a_tri = 1.0 / 6.0;
            const Real b_tri = 2.0 / 3.0;
            const Real gz = 0.577350269189626;  // 1/√3
            const Real tri_pts[3][2] = {{a_tri, a_tri}, {b_tri, a_tri}, {a_tri, b_tri}};
            const Real z_pts[2] = {-gz, gz};
            const Real gp_weight = 1.0 / 6.0;  // Triangle weight × z weight (1/6 × 1)

            // Launch GPU kernel
            Kokkos::parallel_for("Wedge6_ElementForces", num_elems, KOKKOS_LAMBDA(const int e) {
                const int nnodes = 6;
                const int ndof_elem = 18;

                // Stack arrays
                Real elem_coords[18];  // 6 nodes × 3 coords
                Real elem_disp[18];
                Real elem_force[18];

                // Initialize
                for (int i = 0; i < ndof_elem; ++i) {
                    elem_force[i] = 0.0;
                }

                // Extract element data
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    elem_coords[n*3 + 0] = coords_d(node, 0);
                    elem_coords[n*3 + 1] = coords_d(node, 1);
                    elem_coords[n*3 + 2] = coords_d(node, 2);

                    elem_disp[n*3 + 0] = disp_d(node * 3 + 0);
                    elem_disp[n*3 + 1] = disp_d(node * 3 + 1);
                    elem_disp[n*3 + 2] = disp_d(node * 3 + 2);
                }

                // 6-point Gauss integration
                for (int iz = 0; iz < 2; ++iz) {
                    for (int it = 0; it < 3; ++it) {
                        const Real xi = tri_pts[it][0];
                        const Real eta = tri_pts[it][1];
                        const Real zeta = z_pts[iz];
                        const Real w = gp_weight;

                        // ========================================
                        // Wedge6 Shape Derivatives
                        // ========================================
                        Real dN[18];  // 6 nodes × 3 derivatives

                        // Bottom nodes (ζ = -1)
                        dN[0*3 + 0] = -0.5 * (1.0 - zeta);
                        dN[0*3 + 1] = -0.5 * (1.0 - zeta);
                        dN[0*3 + 2] = -0.5 * (1.0 - xi - eta);

                        dN[1*3 + 0] =  0.5 * (1.0 - zeta);
                        dN[1*3 + 1] =  0.0;
                        dN[1*3 + 2] = -0.5 * xi;

                        dN[2*3 + 0] =  0.0;
                        dN[2*3 + 1] =  0.5 * (1.0 - zeta);
                        dN[2*3 + 2] = -0.5 * eta;

                        // Top nodes (ζ = +1)
                        dN[3*3 + 0] = -0.5 * (1.0 + zeta);
                        dN[3*3 + 1] = -0.5 * (1.0 + zeta);
                        dN[3*3 + 2] =  0.5 * (1.0 - xi - eta);

                        dN[4*3 + 0] =  0.5 * (1.0 + zeta);
                        dN[4*3 + 1] =  0.0;
                        dN[4*3 + 2] =  0.5 * xi;

                        dN[5*3 + 0] =  0.0;
                        dN[5*3 + 1] =  0.5 * (1.0 + zeta);
                        dN[5*3 + 2] =  0.5 * eta;

                        // ========================================
                        // Jacobian Matrix
                        // ========================================
                        Real J[9] = {0};
                        for (int n = 0; n < nnodes; ++n) {
                            const Real x = elem_coords[n*3 + 0];
                            const Real y = elem_coords[n*3 + 1];
                            const Real z = elem_coords[n*3 + 2];

                            J[0] += dN[n*3 + 0] * x;
                            J[1] += dN[n*3 + 1] * x;
                            J[2] += dN[n*3 + 2] * x;

                            J[3] += dN[n*3 + 0] * y;
                            J[4] += dN[n*3 + 1] * y;
                            J[5] += dN[n*3 + 2] * y;

                            J[6] += dN[n*3 + 0] * z;
                            J[7] += dN[n*3 + 1] * z;
                            J[8] += dN[n*3 + 2] * z;
                        }

                        // Determinant
                        const Real detJ = J[0]*(J[4]*J[8] - J[5]*J[7])
                                        - J[1]*(J[3]*J[8] - J[5]*J[6])
                                        + J[2]*(J[3]*J[7] - J[4]*J[6]);

                        if (detJ <= 0.0) continue;

                        // Inverse Jacobian
                        const Real invDetJ = 1.0 / detJ;
                        Real J_inv[9];
                        J_inv[0] = (J[4]*J[8] - J[5]*J[7]) * invDetJ;
                        J_inv[1] = (J[2]*J[7] - J[1]*J[8]) * invDetJ;
                        J_inv[2] = (J[1]*J[5] - J[2]*J[4]) * invDetJ;
                        J_inv[3] = (J[5]*J[6] - J[3]*J[8]) * invDetJ;
                        J_inv[4] = (J[0]*J[8] - J[2]*J[6]) * invDetJ;
                        J_inv[5] = (J[2]*J[3] - J[0]*J[5]) * invDetJ;
                        J_inv[6] = (J[3]*J[7] - J[4]*J[6]) * invDetJ;
                        J_inv[7] = (J[1]*J[6] - J[0]*J[7]) * invDetJ;
                        J_inv[8] = (J[0]*J[4] - J[1]*J[3]) * invDetJ;

                        // ========================================
                        // Global derivatives
                        // ========================================
                        Real dN_dx[18];
                        for (int n = 0; n < nnodes; ++n) {
                            dN_dx[n*3 + 0] = J_inv[0]*dN[n*3+0] + J_inv[1]*dN[n*3+1] + J_inv[2]*dN[n*3+2];
                            dN_dx[n*3 + 1] = J_inv[3]*dN[n*3+0] + J_inv[4]*dN[n*3+1] + J_inv[5]*dN[n*3+2];
                            dN_dx[n*3 + 2] = J_inv[6]*dN[n*3+0] + J_inv[7]*dN[n*3+1] + J_inv[8]*dN[n*3+2];
                        }

                        // ========================================
                        // Strain (Voigt notation)
                        // ========================================
                        Real strain[6] = {0};
                        for (int n = 0; n < nnodes; ++n) {
                            const Real ux = elem_disp[n*3 + 0];
                            const Real uy = elem_disp[n*3 + 1];
                            const Real uz = elem_disp[n*3 + 2];

                            strain[0] += dN_dx[n*3 + 0] * ux;
                            strain[1] += dN_dx[n*3 + 1] * uy;
                            strain[2] += dN_dx[n*3 + 2] * uz;
                            strain[3] += dN_dx[n*3 + 1] * ux + dN_dx[n*3 + 0] * uy;
                            strain[4] += dN_dx[n*3 + 2] * uy + dN_dx[n*3 + 1] * uz;
                            strain[5] += dN_dx[n*3 + 2] * ux + dN_dx[n*3 + 0] * uz;
                        }

                        // ========================================
                        // Stress (linear elastic)
                        // ========================================
                        Real stress[6];
                        const Real trace = strain[0] + strain[1] + strain[2];
                        stress[0] = lambda * trace + 2.0 * mu * strain[0];
                        stress[1] = lambda * trace + 2.0 * mu * strain[1];
                        stress[2] = lambda * trace + 2.0 * mu * strain[2];
                        stress[3] = mu * strain[3];
                        stress[4] = mu * strain[4];
                        stress[5] = mu * strain[5];

                        // ========================================
                        // Internal forces
                        // ========================================
                        const Real factor = detJ * w;
                        for (int n = 0; n < nnodes; ++n) {
                            elem_force[n*3 + 0] += factor * (dN_dx[n*3+0]*stress[0] +
                                                              dN_dx[n*3+1]*stress[3] +
                                                              dN_dx[n*3+2]*stress[5]);
                            elem_force[n*3 + 1] += factor * (dN_dx[n*3+1]*stress[1] +
                                                              dN_dx[n*3+0]*stress[3] +
                                                              dN_dx[n*3+2]*stress[4]);
                            elem_force[n*3 + 2] += factor * (dN_dx[n*3+2]*stress[2] +
                                                              dN_dx[n*3+1]*stress[4] +
                                                              dN_dx[n*3+0]*stress[5]);
                        }
                    }  // Triangle loop
                }  // ζ loop

                // ========================================
                // Assemble to global
                // ========================================
                for (int n = 0; n < nnodes; ++n) {
                    const Index node = conn_d(e * nnodes + n);
                    for (int d = 0; d < 3; ++d) {
                        const Index global_dof = node * 3 + d;
                        Kokkos::atomic_add(&f_int_d(global_dof), elem_force[n*3 + d]);
                    }
                }
            });

            Kokkos::fence("Wedge6_ElementForces_complete");
            used_gpu_kernel = true;

            continue;  // Skip CPU fallback for Wedge6
        }

        // CPU fallback for other element types
        NXS_LOG_INFO("Using CPU fallback for element type {} ({} elements)",
                     static_cast<int>(props.type), num_elems);

        // Sync to host for CPU computation
        displacement_.sync_host();
        force_internal_.sync_host();
        auto disp_h = displacement_.view_host();
        auto f_int_h = force_internal_.view_host();

        std::vector<Real> elem_coords(nodes_per_elem * 3);
        std::vector<Real> elem_disp(dof_per_elem);
        std::vector<Real> elem_strain(6, 0.0);  // Voigt notation: [εxx, εyy, εzz, γxy, γyz, γxz]
        std::vector<Real> elem_stress(6, 0.0);  // Voigt notation: [σxx, σyy, σzz, τxy, τyz, τxz]
        std::vector<Real> elem_force(dof_per_elem);

        // Loop over elements
        for (std::size_t e = 0; e < num_elems; ++e) {
            const std::size_t conn_offset = e * nodes_per_elem;

            // Extract element coordinates and displacements
            for (std::size_t n = 0; n < nodes_per_elem; ++n) {
                const Index node = group.connectivity[conn_offset + n];
                elem_coords[n * 3 + 0] = coords.at(node, 0);
                elem_coords[n * 3 + 1] = coords.at(node, 1);
                elem_coords[n * 3 + 2] = coords.at(node, 2);

                for (Index d = 0; d < props.num_dof_per_node; ++d) {
                    const Index global_dof = node * dof_per_node_ + d;
                    const Index local_dof = n * props.num_dof_per_node + d;
                    elem_disp[local_dof] = disp_h(global_dof);
                }
            }

            // Compute internal forces using element's specified integration scheme
            // Each element type can define its own optimal integration strategy
            std::fill(elem_force.begin(), elem_force.end(), 0.0);

            // Material constants
            const Real lambda = group.material.E * group.material.nu /
                              ((1.0 + group.material.nu) * (1.0 - 2.0 * group.material.nu));
            const Real mu = group.material.E / (2.0 * (1.0 + group.material.nu));

            // Get element's integration points and weights
            const int num_gp = props.num_gauss_points;
            std::vector<Real> gp_coords(num_gp * 3);
            std::vector<Real> gp_weights(num_gp);
            elem->gauss_quadrature(gp_coords.data(), gp_weights.data());

            std::vector<Real> B(6 * dof_per_elem);

            // Loop over integration points
            for (int gp_idx = 0; gp_idx < num_gp; ++gp_idx) {
                // Get integration point coordinates
                Real xi[3] = {
                    gp_coords[gp_idx * 3 + 0],
                    gp_coords[gp_idx * 3 + 1],
                    gp_coords[gp_idx * 3 + 2]
                };
                const Real weight = gp_weights[gp_idx];

                // Compute B-matrix at this integration point
                elem->strain_displacement_matrix(xi, elem_coords.data(), B.data());

                // Compute Jacobian determinant
                Real J[9];
                elem->jacobian(xi, elem_coords.data(), J);
                const Real detJ = J[0] * (J[4]*J[8] - J[5]*J[7])
                                - J[1] * (J[3]*J[8] - J[5]*J[6])
                                + J[2] * (J[3]*J[7] - J[4]*J[6]);

                // Compute strain: ε = B * u
                std::fill(elem_strain.begin(), elem_strain.end(), 0.0);
                for (std::size_t i = 0; i < 6; ++i) {
                    for (std::size_t j = 0; j < dof_per_elem; ++j) {
                        elem_strain[i] += B[i * dof_per_elem + j] * elem_disp[j];
                    }
                }

                // Compute stress: σ = C * ε
                elem_stress[0] = (lambda + 2.0 * mu) * elem_strain[0] + lambda * (elem_strain[1] + elem_strain[2]);
                elem_stress[1] = (lambda + 2.0 * mu) * elem_strain[1] + lambda * (elem_strain[0] + elem_strain[2]);
                elem_stress[2] = (lambda + 2.0 * mu) * elem_strain[2] + lambda * (elem_strain[0] + elem_strain[1]);
                elem_stress[3] = mu * elem_strain[3];
                elem_stress[4] = mu * elem_strain[4];
                elem_stress[5] = mu * elem_strain[5];

                // Accumulate force: f += B^T * σ * detJ * weight
                for (std::size_t j = 0; j < dof_per_elem; ++j) {
                    for (std::size_t i = 0; i < 6; ++i) {
                        elem_force[j] += B[i * dof_per_elem + j] * elem_stress[i] * detJ * weight;
                    }
                }
            }

            // NOTE: Hourglass control disabled for bending-dominated problems
            // Standard hex8 with 1-point integration experiences:
            // - Volumetric locking with full (8-point) integration in bending
            // - Hourglass instability with reduced (1-point) integration
            // For pure bending, use higher-order elements (Hex20) or shell elements
            //
            // Hourglass control code kept for reference (currently disabled):
            // if (props.type == physics::ElementType::Hex8) {
            //     const Real mu = group.material.G;
            //     const Real hg_stiffness = 0.0005 * mu;
            //     std::vector<Real> hg_force(dof_per_elem, 0.0);
            //     auto hex8_elem = dynamic_cast<const nxs::fem::Hex8Element*>(elem.get());
            //     if (hex8_elem) {
            //         hex8_elem->hourglass_forces(elem_coords.data(), elem_disp.data(),
            //                                     hg_stiffness, hg_force.data());
            //         for (std::size_t j = 0; j < dof_per_elem; ++j) {
            //             elem_force[j] += hg_force[j];
            //         }
            //     }
            // }

            // Assemble into global force vector
            for (std::size_t n = 0; n < nodes_per_elem; ++n) {
                const Index node = group.connectivity[conn_offset + n];
                for (Index d = 0; d < props.num_dof_per_node; ++d) {
                    const Index global_dof = node * dof_per_node_ + d;
                    const Index local_dof = n * props.num_dof_per_node + d;
                    f_int_h(global_dof) += elem_force[local_dof];
                }
            }
        }
    }

    // Mark where data was modified
    if (used_gpu_kernel) {
        force_internal_.modify_device();
    } else {
        force_internal_.modify_host();
    }
}

// ============================================================================
// Boundary Conditions
// ============================================================================

void FEMSolver::apply_force_boundary_conditions(Real time)
{
    auto f_ext_h = force_external_.view_host();

    // Apply force, pressure, and other non-essential BCs
    for (const auto& bc : boundary_conditions_) {
        if (bc.type != BCType::Force && bc.type != BCType::Pressure) {
            continue;  // Skip essential BCs
        }

        // Compute time-dependent value
        Real value = bc.value;
        if (bc.time_function != nullptr) {
            value *= bc.time_function(time);
        }

        // Apply BC to all specified nodes
        for (const Index node : bc.nodes) {
            const Index dof = node * dof_per_node_ + bc.dof;

            switch (bc.type) {
                case BCType::Force:
                    f_ext_h(dof) += value;
                    break;

                case BCType::Pressure:
                    // Pressure BC needs element face area - not implemented yet
                    NXS_LOG_WARN("Pressure BC not yet implemented");
                    break;

                default:
                    break;
            }
        }
    }

    force_external_.modify_host();
}

void FEMSolver::apply_displacement_boundary_conditions(Real time)
{
    auto disp_h = displacement_.view_host();
    auto vel_h = velocity_.view_host();
    auto acc_h = acceleration_.view_host();

    // Apply essential (displacement) BCs after time integration
    for (const auto& bc : boundary_conditions_) {
        if (bc.type != BCType::Displacement && bc.type != BCType::Velocity &&
            bc.type != BCType::Acceleration) {
            continue;  // Skip non-essential BCs
        }

        // Compute time-dependent value
        Real value = bc.value;
        if (bc.time_function != nullptr) {
            value *= bc.time_function(time);
        }

        // Apply BC to all specified nodes
        for (const Index node : bc.nodes) {
            const Index dof = node * dof_per_node_ + bc.dof;

            switch (bc.type) {
                case BCType::Displacement:
                    disp_h(dof) = value;
                    vel_h(dof) = 0.0;
                    acc_h(dof) = 0.0;
                    break;

                case BCType::Velocity:
                    vel_h(dof) = value;
                    break;

                case BCType::Acceleration:
                    acc_h(dof) = value;
                    break;

                default:
                    break;
            }
        }
    }

    displacement_.modify_host();
    velocity_.modify_host();
    acceleration_.modify_host();
}

void FEMSolver::apply_boundary_conditions(Real time)
{
    // Legacy method - applies all BCs
    // For backward compatibility
    apply_force_boundary_conditions(time);
    apply_displacement_boundary_conditions(time);
}

// ============================================================================
// Configuration
// ============================================================================

void FEMSolver::add_element_group(const std::string& name,
                                  physics::ElementType type,
                                  const std::vector<Index>& element_ids,
                                  const std::vector<Index>& connectivity,
                                  const physics::MaterialProperties& material)
{
    ElementGroup group;
    group.name = name;
    group.type = type;
    group.element_ids = element_ids;
    group.connectivity = connectivity;
    group.material = material;

    // Create element instance
    switch (type) {
        case physics::ElementType::Hex8:
            group.element = std::make_shared<Hex8Element>();
            break;
        case physics::ElementType::Hex20:
            group.element = std::make_shared<Hex20Element>();
            break;
        case physics::ElementType::Tet4:
            group.element = std::make_shared<Tet4Element>();
            break;
        case physics::ElementType::Tet10:
            group.element = std::make_shared<Tet10Element>();
            break;
        case physics::ElementType::Wedge6:
            group.element = std::make_shared<Wedge6Element>();
            break;
        case physics::ElementType::Shell4:
            group.element = std::make_shared<Shell4Element>();
            break;
        case physics::ElementType::Beam2:
            group.element = std::make_shared<Beam2Element>();
            break;
        default:
            throw NotImplementedError("Element type not supported");
    }

    element_groups_.push_back(std::move(group));

    NXS_LOG_INFO("Added element group '{}' ({} elements)",
                 name, element_ids.size());
}

void FEMSolver::add_boundary_condition(const BoundaryCondition& bc)
{
    boundary_conditions_.push_back(bc);

    std::string bc_type_str;
    switch (bc.type) {
        case BCType::Displacement: bc_type_str = "Displacement"; break;
        case BCType::Velocity: bc_type_str = "Velocity"; break;
        case BCType::Acceleration: bc_type_str = "Acceleration"; break;
        case BCType::Force: bc_type_str = "Force"; break;
        case BCType::Pressure: bc_type_str = "Pressure"; break;
    }

    NXS_LOG_INFO("Added BC: {} = {} on {} nodes (DOF {})",
                 bc_type_str, bc.value, bc.nodes.size(), bc.dof);
}

// ============================================================================
// Field Exchange
// ============================================================================

std::vector<std::string> FEMSolver::provided_fields() const
{
    return {"displacement", "velocity", "acceleration", "stress", "strain"};
}

std::vector<std::string> FEMSolver::required_fields() const
{
    return {};  // FEM doesn't require fields from other modules
}

void FEMSolver::export_field(const std::string& field_name,
                             std::vector<Real>& data) const
{
    if (field_name == "displacement") {
        const_cast<Kokkos::DualView<Real*>&>(displacement_).sync_host();
        auto view_h = displacement_.view_host();
        data.resize(ndof_);
        for (std::size_t i = 0; i < ndof_; ++i) {
            data[i] = view_h(i);
        }
    } else if (field_name == "velocity") {
        const_cast<Kokkos::DualView<Real*>&>(velocity_).sync_host();
        auto view_h = velocity_.view_host();
        data.resize(ndof_);
        for (std::size_t i = 0; i < ndof_; ++i) {
            data[i] = view_h(i);
        }
    } else if (field_name == "acceleration") {
        const_cast<Kokkos::DualView<Real*>&>(acceleration_).sync_host();
        auto view_h = acceleration_.view_host();
        data.resize(ndof_);
        for (std::size_t i = 0; i < ndof_; ++i) {
            data[i] = view_h(i);
        }
    } else if (field_name == "force_internal") {
        const_cast<Kokkos::DualView<Real*>&>(force_internal_).sync_host();
        auto view_h = force_internal_.view_host();
        data.resize(ndof_);
        for (std::size_t i = 0; i < ndof_; ++i) {
            data[i] = view_h(i);
        }
    } else if (field_name == "force_external") {
        const_cast<Kokkos::DualView<Real*>&>(force_external_).sync_host();
        auto view_h = force_external_.view_host();
        data.resize(ndof_);
        for (std::size_t i = 0; i < ndof_; ++i) {
            data[i] = view_h(i);
        }
    } else {
        throw InvalidArgumentError("Unknown field: " + field_name);
    }
}

void FEMSolver::import_field(const std::string& field_name,
                             const std::vector<Real>& data)
{
    if (field_name == "displacement") {
        auto view_h = displacement_.view_host();
        for (std::size_t i = 0; i < std::min(ndof_, data.size()); ++i) {
            view_h(i) = data[i];
        }
        displacement_.modify_host();
    } else if (field_name == "velocity") {
        auto view_h = velocity_.view_host();
        for (std::size_t i = 0; i < std::min(ndof_, data.size()); ++i) {
            view_h(i) = data[i];
        }
        velocity_.modify_host();
    } else {
        throw InvalidArgumentError("Cannot import field: " + field_name);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void FEMSolver::zero_forces()
{
    // Note: force_internal is zeroed in compute_internal_forces() on device
    // Here we only zero force_external on host (it's modified on host by BCs)
    force_external_.sync_host();
    auto f_ext_h = force_external_.view_host();

    for (std::size_t i = 0; i < ndof_; ++i) {
        f_ext_h(i) = 0.0;
    }

    force_external_.modify_host();

    // Don't touch force_internal here - it will be zeroed and computed on device
}

void FEMSolver::apply_body_forces()
{
    if (!gravity_enabled_ && !body_force_enabled_) {
        return;  // No body forces to apply
    }

    // Get host views
    auto f_ext_h = force_external_.view_host();
    auto mass_h = mass_.view_host();

    // Apply gravity: f_gravity = m * g
    // For lumped mass matrix, each DOF has its own mass
    // Gravity acts on displacement DOFs (x, y, z)
    if (gravity_enabled_) {
        for (std::size_t node = 0; node < num_nodes_; ++node) {
            for (Index d = 0; d < dof_per_node_ && d < 3; ++d) {
                Index dof = node * dof_per_node_ + d;
                // f = m * g (using lumped mass at this DOF)
                f_ext_h(dof) += mass_h(dof) * gravity_[d];
            }
        }
    }

    // Apply body force: f_body = b * V (force per unit volume × nodal volume)
    // For lumped mass: V_node = m_node / ρ
    // Since we don't have density per node, we use: f = b * (m / ρ)
    // This is typically applied via consistent element integration,
    // but for simplicity with lumped mass, we approximate using mass ratio
    if (body_force_enabled_) {
        // Get average density from first element group
        Real density = 1000.0;  // Default
        if (!element_groups_.empty()) {
            density = element_groups_[0].material.density;
        }

        for (std::size_t node = 0; node < num_nodes_; ++node) {
            for (Index d = 0; d < dof_per_node_ && d < 3; ++d) {
                Index dof = node * dof_per_node_ + d;
                // f = b * V = b * (m / ρ)
                Real volume = mass_h(dof) / density;
                f_ext_h(dof) += body_force_[d] * volume;
            }
        }
    }

    force_external_.modify_host();
}

Real FEMSolver::compute_element_size(const ElementGroup& group, Index elem_id) const
{
    // Find element index in group
    auto it = std::find(group.element_ids.begin(), group.element_ids.end(), elem_id);
    if (it == group.element_ids.end()) {
        return 1.0;  // Default if not found
    }

    const std::size_t e = std::distance(group.element_ids.begin(), it);
    const auto props = group.element->properties();
    const std::size_t nodes_per_elem = props.num_nodes;
    const std::size_t conn_offset = e * nodes_per_elem;

    // Get node coordinates
    const auto& coords = mesh_->coordinates();
    std::vector<Real> elem_coords(nodes_per_elem * 3);

    for (std::size_t n = 0; n < nodes_per_elem; ++n) {
        const Index node = group.connectivity[conn_offset + n];
        elem_coords[n * 3 + 0] = coords.at(node, 0);
        elem_coords[n * 3 + 1] = coords.at(node, 1);
        elem_coords[n * 3 + 2] = coords.at(node, 2);
    }

    // Return characteristic length
    return group.element->characteristic_length(elem_coords.data());
}

Real FEMSolver::compute_wave_speed(const physics::MaterialProperties& mat) const
{
    // Compute elastic wave speed: c = sqrt(E/ρ) for 1D
    // For 3D: c = sqrt((λ + 2μ)/ρ) where λ and μ are Lamé parameters
    // Simplification: c = sqrt(E/ρ)
    return std::sqrt(mat.E / mat.density);
}

} // namespace fem
} // namespace nxs
