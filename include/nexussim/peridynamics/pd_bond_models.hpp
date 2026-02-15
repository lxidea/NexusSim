#pragma once

/**
 * @file pd_bond_models.hpp
 * @brief Enhanced bond-based PD models beyond PMB
 *
 * Models:
 * - EnergyBased: critical energy release rate failure
 * - Microplastic: bilinear force-stretch with plasticity
 * - Viscoelastic: standard linear solid bond model
 * - ShortRange: repulsive penalty for contact/compression
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>

namespace nxs {
namespace pd {

// ============================================================================
// Bond model type
// ============================================================================

enum class BondModelType {
    EnergyBased,
    Microplastic,
    Viscoelastic,
    ShortRange
};

// ============================================================================
// Bond history — per-bond state variables
// ============================================================================

struct BondModelParams {
    BondModelType type = BondModelType::EnergyBased;

    // Energy-based failure
    Real Gc = 22000.0;             // Fracture energy (J/m^2)

    // Microplastic
    Real s_yield = 0.001;          // Yield stretch
    Real hardening_ratio = 0.1;    // beta: post-yield slope ratio (0=perfect plastic, 1=elastic)

    // Viscoelastic (standard linear solid)
    Real c_inf_ratio = 0.5;        // Long-term modulus ratio c_inf/c
    Real c1_ratio = 0.5;           // Prony series coefficient ratio c1/c
    Real tau = 1e-5;               // Relaxation time (s)

    // Short-range repulsion
    Real k_rep = 1e12;             // Repulsive stiffness
    Real r_min_ratio = 0.9;        // Minimum distance ratio (r_min = ratio * |xi|)
};

// ============================================================================
// PDEnhancedBondForce
// ============================================================================

class PDEnhancedBondForce {
public:
    PDEnhancedBondForce() = default;

    void initialize(const std::vector<PDMaterial>& materials,
                    const BondModelParams& params) {
        num_materials_ = materials.size();
        params_ = params;

        c_ = PDScalarView("enh_c", num_materials_);
        s_critical_ = PDScalarView("enh_s_critical", num_materials_);

        auto c_host = Kokkos::create_mirror_view(c_);
        auto s_crit_host = Kokkos::create_mirror_view(s_critical_);

        for (Index i = 0; i < num_materials_; ++i) {
            c_host(i) = materials[i].c;
            s_crit_host(i) = materials[i].s_critical;
        }

        Kokkos::deep_copy(c_, c_host);
        Kokkos::deep_copy(s_critical_, s_crit_host);
    }

    /**
     * @brief Allocate bond history storage after neighbor list is built
     *
     * history(*,0) = plastic_stretch (microplastic) / energy_released (energy-based)
     * history(*,1) = viscous_stretch (viscoelastic)
     * history(*,2) = reserved
     */
    void allocate_history(Index total_bonds) {
        bond_history_ = Kokkos::View<Real*[3]>("bond_history", total_bonds);
        Kokkos::deep_copy(bond_history_, 0.0);
    }

    /**
     * @brief Compute enhanced bond forces
     */
    void compute_forces(PDParticleSystem& particles, PDNeighborList& neighbors,
                        Real dt) {
        particles.zero_forces();

        auto x = particles.x();
        auto u = particles.u();
        auto f = particles.f();
        auto volume = particles.volume();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto c = c_;
        auto history = bond_history_;
        auto model_type = params_.type;
        auto s_yield = params_.s_yield;
        auto beta = params_.hardening_ratio;
        auto c_inf_ratio = params_.c_inf_ratio;
        auto c1_ratio = params_.c1_ratio;
        auto tau = params_.tau;
        auto k_rep = params_.k_rep;
        auto r_min_ratio = params_.r_min_ratio;
        auto Gc = params_.Gc;

        Index num_particles = particles.num_particles();

        Kokkos::parallel_for("compute_enhanced_bond_forces", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);
                Index mat_i = material_id(i);
                Real c_i = c(mat_i);

                Real fi[3] = {0.0, 0.0, 0.0};

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);
                    Real Vj = volume(j);

                    // Compute stretch
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };
                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };
                    Real xi_eta[3] = {
                        xi[0] + eta[0],
                        xi[1] + eta[1],
                        xi[2] + eta[2]
                    };
                    Real xi_eta_len = Kokkos::sqrt(
                        xi_eta[0] * xi_eta[0] +
                        xi_eta[1] * xi_eta[1] +
                        xi_eta[2] * xi_eta[2]
                    );
                    Real s = (xi_eta_len - xi_len) / xi_len;

                    // Unit direction
                    Real inv_len = 1.0 / (xi_eta_len + 1e-20);
                    Real e[3] = {
                        xi_eta[0] * inv_len,
                        xi_eta[1] * inv_len,
                        xi_eta[2] * inv_len
                    };

                    Real t = 0.0;

                    switch (model_type) {
                        case BondModelType::EnergyBased: {
                            // Elastic force
                            t = c_i * s * w;

                            // Accumulate energy in history[0]
                            // dW = 0.5 * c * s^2 * |xi| * w * Vj (per bond per step)
                            Real dW = 0.5 * c_i * s * s * xi_len * w * Vj;
                            Real W_old = history(bond_idx, 0);
                            Real W_new = W_old + Kokkos::fabs(dW);

                            // Critical energy per bond: gc_bond = Gc / (sum |xi| * Vj)
                            // Approximate: gc_bond ~ Gc * xi_len
                            Real gc_bond = Gc * xi_len;

                            if (W_new >= gc_bond) {
                                bond_intact(bond_idx) = false;
                                t = 0.0;
                            }
                            history(bond_idx, 0) = W_new;
                            break;
                        }
                        case BondModelType::Microplastic: {
                            Real s_p = history(bond_idx, 0); // plastic stretch

                            Real s_elastic = s - s_p;

                            if (s_elastic > s_yield) {
                                // Plastic loading
                                Real ds_p = s_elastic - s_yield;
                                s_p += (1.0 - beta) * ds_p;
                                s_elastic = s_yield + beta * ds_p;
                                history(bond_idx, 0) = s_p;
                            } else if (s_elastic < -s_yield) {
                                // Compressive yielding
                                Real ds_p = s_elastic + s_yield;
                                s_p += (1.0 - beta) * ds_p;
                                s_elastic = -s_yield + beta * ds_p;
                                history(bond_idx, 0) = s_p;
                            }

                            t = c_i * s_elastic * w;
                            break;
                        }
                        case BondModelType::Viscoelastic: {
                            // Standard linear solid: c(t) = c_inf + c1 * exp(-t/tau)
                            Real c_inf = c_inf_ratio * c_i;
                            Real c1 = c1_ratio * c_i;

                            Real s_v = history(bond_idx, 1); // viscous internal variable

                            // Update viscous stretch with exponential decay
                            Real exp_dt = Kokkos::exp(-dt / tau);
                            s_v = exp_dt * s_v + (1.0 - exp_dt) * s;
                            history(bond_idx, 1) = s_v;

                            // Force: t = c_inf * s + c1 * (s - s_v)
                            t = (c_inf * s + c1 * (s - s_v)) * w;
                            break;
                        }
                        case BondModelType::ShortRange: {
                            // Repulsive force only for compression
                            Real r_min = r_min_ratio * xi_len;
                            if (xi_eta_len < r_min) {
                                Real penetration = xi_eta_len - r_min; // negative
                                t = k_rep * penetration / xi_len;
                                // No attraction
                            } else {
                                t = 0.0;
                            }
                            break;
                        }
                    }

                    fi[0] += t * e[0] * Vj;
                    fi[1] += t * e[1] * Vj;
                    fi[2] += t * e[2] * Vj;
                }

                Kokkos::atomic_add(&f(i, 0), fi[0]);
                Kokkos::atomic_add(&f(i, 1), fi[1]);
                Kokkos::atomic_add(&f(i, 2), fi[2]);
            });
    }

    // Accessors
    Kokkos::View<Real*[3]>& bond_history() { return bond_history_; }
    const BondModelParams& params() const { return params_; }

private:
    Index num_materials_ = 0;
    BondModelParams params_;

    PDScalarView c_;
    PDScalarView s_critical_;
    Kokkos::View<Real*[3]> bond_history_;
};

} // namespace pd
} // namespace nxs
