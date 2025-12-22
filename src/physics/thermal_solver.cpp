/**
 * @file thermal_solver.cpp
 * @brief Thermal solver implementation
 */

#include <nexussim/physics/thermal_solver.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace nxs {
namespace physics {

// Stefan-Boltzmann constant (W/m²·K⁴)
constexpr Real STEFAN_BOLTZMANN = 5.670374419e-8;

ThermalSolver::ThermalSolver(const std::string& name)
    : PhysicsModule(name, Type::Custom)
    , num_nodes_(0)
    , reference_temperature_(293.15)    // Room temperature (K)
    , initial_temperature_(293.15)
    , cfl_factor_(0.4)                  // Conservative for explicit thermal
    , adiabatic_heating_enabled_(false)
    , min_element_size_(0.01)           // Default 1cm
    , max_temperature_(293.15)
    , min_temperature_(293.15)
    , avg_temperature_(293.15)
{}

void ThermalSolver::initialize(std::shared_ptr<Mesh> mesh,
                               std::shared_ptr<State> state) {
    mesh_ = mesh;
    state_ = state;

    num_nodes_ = mesh->num_nodes();

    // Allocate temperature arrays
    temperature_ = Kokkos::DualView<Real*>("temperature", num_nodes_);
    temperature_old_ = Kokkos::DualView<Real*>("temperature_old", num_nodes_);
    heat_rate_ = Kokkos::DualView<Real*>("heat_rate", num_nodes_);
    heat_source_ = Kokkos::DualView<Real*>("heat_source", num_nodes_);

    // Initialize to initial temperature
    auto T_host = temperature_.view_host();
    auto T_old_host = temperature_old_.view_host();
    auto hr_host = heat_rate_.view_host();
    auto hs_host = heat_source_.view_host();

    for (size_t i = 0; i < num_nodes_; ++i) {
        T_host(i) = initial_temperature_;
        T_old_host(i) = initial_temperature_;
        hr_host(i) = 0.0;
        hs_host(i) = 0.0;
    }

    temperature_.modify_host();
    temperature_old_.modify_host();
    heat_rate_.modify_host();
    heat_source_.modify_host();

    // Sync to device
    temperature_.sync_device();
    temperature_old_.sync_device();
    heat_rate_.sync_device();
    heat_source_.sync_device();

    // Compute minimum element size for stability
    min_element_size_ = compute_min_element_size();

    initialized_ = true;
}

void ThermalSolver::finalize() {
    // Nothing to clean up for now
}

Real ThermalSolver::compute_stable_dt() const {
    // Thermal stability: dt < cfl * h² / (2*dim*α)
    // where α = k/(ρ*c) is thermal diffusivity

    Real alpha = default_material_.diffusivity();
    Real h = min_element_size_;

    // 3D stability criterion
    Real dt_stable = cfl_factor_ * h * h / (6.0 * alpha);

    return dt_stable;
}

void ThermalSolver::step(Real dt) {
    // Save old temperature
    auto T_host = temperature_.view_host();
    auto T_old_host = temperature_old_.view_host();

    for (size_t i = 0; i < num_nodes_; ++i) {
        T_old_host(i) = T_host(i);
    }
    temperature_old_.modify_host();

    // Compute heat conduction
    compute_heat_conduction();

    // Update temperature
    update_temperature(dt);

    // Apply boundary conditions
    apply_boundary_conditions();

    // Update statistics
    max_temperature_ = T_host(0);
    min_temperature_ = T_host(0);
    Real sum = 0.0;

    for (size_t i = 0; i < num_nodes_; ++i) {
        max_temperature_ = std::max(max_temperature_, T_host(i));
        min_temperature_ = std::min(min_temperature_, T_host(i));
        sum += T_host(i);
    }
    avg_temperature_ = sum / num_nodes_;

    // Sync to device
    temperature_.sync_device();
}

void ThermalSolver::compute_heat_conduction() {
    // Simple Laplacian approximation using mesh connectivity
    // For each node: ∇²T ≈ (1/n) * Σ(T_j - T_i) / h²

    auto T_host = temperature_.view_host();
    auto hr_host = heat_rate_.view_host();
    auto hs_host = heat_source_.view_host();

    const Real k = default_material_.conductivity;
    const Real rho = default_material_.density;
    const Real c = default_material_.specific_heat;
    const Real h2 = min_element_size_ * min_element_size_;

    // Zero heat rate
    for (size_t i = 0; i < num_nodes_; ++i) {
        hr_host(i) = 0.0;
    }

    // For each element block, compute heat flow between connected nodes
    for (size_t b = 0; b < mesh_->num_element_blocks(); ++b) {
        const auto& block = mesh_->element_block(b);
        size_t num_elems = block.num_elements();
        size_t nodes_per_elem = block.num_nodes_per_elem;

        for (size_t e = 0; e < num_elems; ++e) {
            auto elem_nodes = block.element_nodes(e);

            // Compute average temperature in element
            Real T_avg = 0.0;
            for (size_t n = 0; n < nodes_per_elem; ++n) {
                T_avg += T_host(elem_nodes[n]);
            }
            T_avg /= nodes_per_elem;

            // Distribute heat based on temperature difference
            for (size_t n = 0; n < nodes_per_elem; ++n) {
                Index node_i = elem_nodes[n];
                Real T_i = T_host(node_i);

                // Heat flux from neighbors (simplified)
                Real laplacian = (T_avg - T_i) * (nodes_per_elem - 1) / h2;

                // Heat rate: dT/dt = (k/ρc)*∇²T + Q/(ρc)
                hr_host(node_i) += k * laplacian;
            }
        }
    }

    // Add heat sources
    for (size_t i = 0; i < num_nodes_; ++i) {
        hr_host(i) += hs_host(i);
        // Normalize by ρ*c
        hr_host(i) /= (rho * c);
    }

    heat_rate_.modify_host();
}

void ThermalSolver::update_temperature(Real dt) {
    auto T_host = temperature_.view_host();
    auto hr_host = heat_rate_.view_host();

    // Forward Euler: T^{n+1} = T^n + dt * dT/dt
    for (size_t i = 0; i < num_nodes_; ++i) {
        T_host(i) += dt * hr_host(i);
    }

    temperature_.modify_host();
}

void ThermalSolver::apply_boundary_conditions() {
    auto T_host = temperature_.view_host();

    for (const auto& bc : boundary_conditions_) {
        switch (bc.type) {
            case ThermalBCType::Temperature:
                // Dirichlet: set temperature directly
                for (Index node : bc.nodes) {
                    T_host(node) = bc.value;
                }
                break;

            case ThermalBCType::HeatFlux:
                // Neumann: heat flux BC (applied in compute_heat_conduction)
                // This is handled differently - skip here
                break;

            case ThermalBCType::Convection:
                // Newton cooling: q = h*(T - T_inf)
                // We apply a pseudo-explicit update
                // T_new = T + dt * h*(T_inf - T) / (rho*c*V/A)
                // Simplified for surface nodes
                for (Index node : bc.nodes) {
                    Real T_surface = T_host(node);
                    Real T_inf = bc.T_ambient;
                    Real h_conv = bc.value;  // Convection coefficient W/m²·K

                    // Simple mixing (surface layer model)
                    Real cooling = h_conv * (T_inf - T_surface);
                    Real rho_c = default_material_.density * default_material_.specific_heat;
                    // Assume surface layer thickness = min_element_size_
                    Real dT = cooling / (rho_c * min_element_size_);
                    // Note: This is per time step, so should be multiplied by dt
                    // For now, apply a small correction
                    T_host(node) += 0.1 * (T_inf - T_surface);
                }
                break;

            case ThermalBCType::Radiation:
                // Stefan-Boltzmann: q = ε*σ*(T^4 - T_inf^4)
                for (Index node : bc.nodes) {
                    Real T_surface = T_host(node);
                    Real T_inf = bc.T_ambient;
                    Real eps = bc.emissivity;

                    Real q_rad = eps * STEFAN_BOLTZMANN *
                                 (std::pow(T_surface, 4) - std::pow(T_inf, 4));

                    Real rho_c = default_material_.density * default_material_.specific_heat;
                    Real dT = -q_rad / (rho_c * min_element_size_);
                    T_host(node) += 0.01 * dT;  // Small correction factor
                }
                break;
        }
    }

    temperature_.modify_host();
}

void ThermalSolver::set_material(const std::vector<Index>& nodes, const ThermalMaterial& mat) {
    for (Index node : nodes) {
        node_materials_[node] = mat;
    }
}

void ThermalSolver::add_heat_source(Index node, Real Q) {
    heat_source_.view_host()(node) += Q;
    heat_source_.modify_host();
}

void ThermalSolver::add_plastic_heating(Index node, Real plastic_work) {
    if (adiabatic_heating_enabled_) {
        Real Q = default_material_.taylor_quinney * plastic_work;
        add_heat_source(node, Q);
    }
}

void ThermalSolver::clear_heat_sources() {
    auto hs_host = heat_source_.view_host();
    for (size_t i = 0; i < num_nodes_; ++i) {
        hs_host(i) = 0.0;
    }
    heat_source_.modify_host();
}

Real ThermalSolver::thermal_strain(Index node) const {
    Real T = temperature_.view_host()(node);
    Real alpha = default_material_.expansion_coeff;

    // Check if node has specific material
    auto it = node_materials_.find(node);
    if (it != node_materials_.end()) {
        alpha = it->second.expansion_coeff;
    }

    return alpha * (T - reference_temperature_);
}

void ThermalSolver::thermal_strain_tensor(Index node, Real* strain) const {
    Real eps_th = thermal_strain(node);

    // Isotropic thermal strain
    strain[0] = eps_th;  // εxx
    strain[1] = eps_th;  // εyy
    strain[2] = eps_th;  // εzz
    strain[3] = 0.0;     // γxy
    strain[4] = 0.0;     // γyz
    strain[5] = 0.0;     // γxz
}

void ThermalSolver::thermal_stress(Index node, const MaterialProperties& mat, Real* stress) const {
    Real T = temperature_.view_host()(node);
    Real alpha = mat.thermal_expansion;
    Real eps_th = alpha * (T - reference_temperature_);

    // Thermal stress for constrained expansion
    Real E = mat.E;
    Real nu = mat.nu;

    // σ_th = -E*α*ΔT / (1-2ν) (hydrostatic for isotropic)
    Real sigma_th = -E * eps_th / (1.0 - 2.0 * nu);

    stress[0] = sigma_th;
    stress[1] = sigma_th;
    stress[2] = sigma_th;
    stress[3] = 0.0;
    stress[4] = 0.0;
    stress[5] = 0.0;
}

Real ThermalSolver::total_thermal_energy() const {
    auto T_host = temperature_.view_host();

    Real energy = 0.0;
    Real rho = default_material_.density;
    Real c = default_material_.specific_heat;

    // Simple approximation: E = ρ*c*V*T
    // Assuming unit volume per node for simplicity
    Real V_node = std::pow(min_element_size_, 3);

    for (size_t i = 0; i < num_nodes_; ++i) {
        energy += rho * c * V_node * T_host(i);
    }

    return energy;
}

Real ThermalSolver::compute_min_element_size() const {
    if (!mesh_ || mesh_->num_element_blocks() == 0) {
        return 0.01;  // Default 1cm
    }

    Real min_h = 1.0e10;

    // Sample first element in first block
    const auto& block = mesh_->element_block(0);
    if (block.num_elements() > 0) {
        auto elem_nodes = block.element_nodes(0);

        // Get coordinates
        auto x0 = mesh_->get_node_coordinates(elem_nodes[0]);
        auto x1 = mesh_->get_node_coordinates(elem_nodes[1]);

        // Compute distance between first two nodes
        Real dx = x1[0] - x0[0];
        Real dy = x1[1] - x0[1];
        Real dz = x1[2] - x0[2];
        Real h = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (h > 1.0e-10) {
            min_h = h;
        }
    }

    return min_h;
}

void ThermalSolver::print_stats(std::ostream& os) const {
    os << "=== Thermal Solver Statistics ===\n";
    os << "Nodes: " << num_nodes_ << "\n";
    os << "Reference temperature: " << reference_temperature_ << " K\n";
    os << "Temperature range: [" << min_temperature_ << ", " << max_temperature_ << "] K\n";
    os << "Average temperature: " << avg_temperature_ << " K\n";
    os << "Material:\n";
    os << "  Density: " << default_material_.density << " kg/m³\n";
    os << "  Specific heat: " << default_material_.specific_heat << " J/kg·K\n";
    os << "  Conductivity: " << default_material_.conductivity << " W/m·K\n";
    os << "  Diffusivity: " << default_material_.diffusivity() << " m²/s\n";
    os << "  Expansion coeff: " << default_material_.expansion_coeff << " 1/K\n";
    os << "Min element size: " << min_element_size_ << " m\n";
    os << "Stable dt: " << compute_stable_dt() << " s\n";
    os << "Adiabatic heating: " << (adiabatic_heating_enabled_ ? "enabled" : "disabled") << "\n";
    os << "Boundary conditions: " << boundary_conditions_.size() << "\n";
    os << "=================================\n";
}

std::vector<std::string> ThermalSolver::provided_fields() const {
    return {"temperature", "thermal_strain"};
}

std::vector<std::string> ThermalSolver::required_fields() const {
    return {};  // No required fields
}

void ThermalSolver::export_field(const std::string& field_name,
                                 std::vector<Real>& data) const {
    if (field_name == "temperature") {
        data.resize(num_nodes_);
        auto T_host = temperature_.view_host();
        for (size_t i = 0; i < num_nodes_; ++i) {
            data[i] = T_host(i);
        }
    } else if (field_name == "thermal_strain") {
        data.resize(num_nodes_);
        for (size_t i = 0; i < num_nodes_; ++i) {
            data[i] = thermal_strain(i);
        }
    }
}

void ThermalSolver::import_field(const std::string& field_name,
                                 const std::vector<Real>& data) {
    if (field_name == "temperature" && data.size() == num_nodes_) {
        auto T_host = temperature_.view_host();
        for (size_t i = 0; i < num_nodes_; ++i) {
            T_host(i) = data[i];
        }
        temperature_.modify_host();
        temperature_.sync_device();
    }
}

} // namespace physics
} // namespace nxs
