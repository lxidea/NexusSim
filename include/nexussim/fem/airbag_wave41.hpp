#pragma once

/**
 * @file airbag_wave41.hpp
 * @brief Wave 41: Airbag Production Hardening — 4 Features
 *
 * Features:
 *   5. AirbagMultiChamber    - Multi-chamber with orifice flow and check valves
 *   6. AirbagGasSpecies      - Multi-species gas tracking with mixing rules
 *   7. AirbagTTF             - Tank test format inflator data import
 *   8. AirbagMembraneDrape   - Gravity-driven membrane draping with fold detection
 *
 * References:
 * - Hallquist et al. (2007) "LS-DYNA Theory Manual - Airbag Models"
 * - Wang & Nefske (1988) "A new CAL3D airbag inflation model"
 * - Marklund & Nilsson (2002) "Optimization of airbag inflation parameters"
 * - Anderson (2003) "Modern Compressible Flow" — isentropic orifice relations
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <cstring>
#include <limits>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Utility functions for airbag Wave 41
// ============================================================================

namespace airbag41_detail {

inline Real clamp(Real x, Real lo, Real hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/// Linear interpolation in a tabulated curve
inline Real interpolate(const std::vector<Real>& x,
                         const std::vector<Real>& y,
                         Real xq)
{
    if (x.empty() || y.empty()) return 0.0;
    if (x.size() == 1) return y[0];

    // Clamp to table bounds
    if (xq <= x.front()) return y.front();
    if (xq >= x.back())  return y.back();

    // Binary search for interval
    auto it = std::lower_bound(x.begin(), x.end(), xq);
    size_t i = static_cast<size_t>(it - x.begin());
    if (i == 0) i = 1;
    if (i >= x.size()) i = x.size() - 1;

    Real t = (xq - x[i-1]) / (x[i] - x[i-1]);
    return y[i-1] + t * (y[i] - y[i-1]);
}

inline Real dot3(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Real norm3(const Real* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

} // namespace airbag41_detail

// ============================================================================
// 5. AirbagMultiChamber — Multi-chamber with orifice flow
// ============================================================================

/**
 * @brief Production multi-chamber airbag model with orifice connections.
 *
 * N chambers connected by orifices. Mass flow through each orifice is computed
 * from isentropic compressible relations with discharge coefficient. Check
 * valves enforce one-way flow.
 */
class AirbagMultiChamber {
public:
    struct Chamber {
        Real V     = 1.0e-3;   ///< Volume [m^3]
        Real P     = 101325.0; ///< Pressure [Pa]
        Real T     = 300.0;    ///< Temperature [K]
        Real mass  = 0.0;      ///< Gas mass [kg]
        Real gamma = 1.4;      ///< Specific heat ratio
        Real R_gas = 287.0;    ///< Specific gas constant [J/(kg*K)]
        int  id    = 0;
    };

    struct Orifice {
        int chamber_1 = 0;      ///< Upstream chamber index
        int chamber_2 = 1;      ///< Downstream chamber index
        Real area     = 1.0e-4; ///< Orifice area [m^2]
        Real Cd       = 0.65;   ///< Discharge coefficient
        bool check_valve = false; ///< One-way flow (1 -> 2 only)
    };

    AirbagMultiChamber() = default;

    /**
     * @brief Add a gas chamber.
     * @return Chamber index
     */
    int add_chamber(Real volume, Real pressure, Real temperature,
                    Real gamma = 1.4, Real R_gas = 287.0)
    {
        Chamber c;
        c.V     = volume;
        c.P     = pressure;
        c.T     = temperature;
        c.gamma = gamma;
        c.R_gas = R_gas;
        c.mass  = pressure * volume / (R_gas * temperature);
        c.id    = static_cast<int>(chambers_.size());
        chambers_.push_back(c);
        return c.id;
    }

    /**
     * @brief Add an orifice connecting two chambers.
     * @return Orifice index
     */
    int add_orifice(int c1, int c2, Real area, Real Cd, bool check_valve = false) {
        Orifice o;
        o.chamber_1   = c1;
        o.chamber_2   = c2;
        o.area        = area;
        o.Cd          = Cd;
        o.check_valve = check_valve;
        orifices_.push_back(o);
        return static_cast<int>(orifices_.size()) - 1;
    }

    /**
     * @brief Advance all chambers by one timestep.
     *
     * Computes orifice mass flow rates, updates chamber masses, recomputes
     * pressures and temperatures using ideal gas law with isentropic assumption.
     *
     * Mass flow: m_dot = Cd * A * p_up * sqrt(2/(gamma*R*T)) * f(p_ratio)
     * where f accounts for choked vs. unchoked flow.
     */
    void step(Real dt) {
        if (chambers_.empty()) return;

        size_t nc = chambers_.size();
        std::vector<Real> dm(nc, 0.0);  // net mass change per chamber
        std::vector<Real> dE(nc, 0.0);  // net energy change per chamber

        for (const auto& orif : orifices_) {
            int i_up = orif.chamber_1;
            int i_dn = orif.chamber_2;
            if (i_up < 0 || i_up >= static_cast<int>(nc)) continue;
            if (i_dn < 0 || i_dn >= static_cast<int>(nc)) continue;

            auto& c_up = chambers_[static_cast<size_t>(i_up)];
            auto& c_dn = chambers_[static_cast<size_t>(i_dn)];

            // Determine flow direction
            Real p_high = c_up.P;
            Real p_low  = c_dn.P;
            int from = i_up;
            int to   = i_dn;

            if (p_low > p_high) {
                if (orif.check_valve) continue; // blocked by check valve
                std::swap(p_high, p_low);
                std::swap(from, to);
            }

            if (p_high <= 0.0) continue;

            auto& c_from = chambers_[static_cast<size_t>(from)];
            Real gam = c_from.gamma;
            Real R   = c_from.R_gas;
            Real T_up = c_from.T;

            // Pressure ratio
            Real p_ratio = p_low / p_high;
            p_ratio = airbag41_detail::clamp(p_ratio, 0.0, 1.0);

            // Critical pressure ratio for choked flow
            Real p_crit = std::pow(2.0 / (gam + 1.0), gam / (gam - 1.0));

            Real m_dot = 0.0;
            if (p_ratio <= p_crit) {
                // Choked flow
                Real factor = gam * std::pow(2.0 / (gam + 1.0),
                              (gam + 1.0) / (gam - 1.0));
                m_dot = orif.Cd * orif.area * p_high
                      * std::sqrt(factor / (R * T_up));
            } else {
                // Subsonic flow
                Real term1 = std::pow(p_ratio, 2.0 / gam);
                Real term2 = std::pow(p_ratio, (gam + 1.0) / gam);
                Real under_sqrt = (2.0 * gam / (gam - 1.0)) * (term1 - term2);
                if (under_sqrt < 0.0) under_sqrt = 0.0;
                m_dot = orif.Cd * orif.area * p_high
                      * std::sqrt(under_sqrt / (R * T_up));
            }

            Real mass_transfer = m_dot * dt;
            // Limit to available mass
            mass_transfer = std::min(mass_transfer,
                                     0.5 * c_from.mass);

            dm[static_cast<size_t>(from)] -= mass_transfer;
            dm[static_cast<size_t>(to)]   += mass_transfer;

            // Energy transfer: enthalpy flux = m_dot * cp * T
            Real cp = gam * R / (gam - 1.0);
            Real energy_transfer = mass_transfer * cp * T_up;
            dE[static_cast<size_t>(from)] -= energy_transfer;
            dE[static_cast<size_t>(to)]   += energy_transfer;
        }

        // Update chambers
        for (size_t i = 0; i < nc; ++i) {
            auto& c = chambers_[i];
            Real old_mass = c.mass;
            c.mass += dm[i];
            if (c.mass < 1.0e-30) c.mass = 1.0e-30;

            // Update temperature from energy balance
            Real cp = c.gamma * c.R_gas / (c.gamma - 1.0);
            Real old_energy = old_mass * cp * c.T;
            Real new_energy = old_energy + dE[i];
            if (new_energy < 0.0) new_energy = 1.0e-10;
            c.T = new_energy / (c.mass * cp);
            if (c.T < 1.0) c.T = 1.0;

            // Update pressure from ideal gas law: P = m * R * T / V
            c.P = c.mass * c.R_gas * c.T / c.V;
        }

        time_ += dt;
    }

    /// Inject mass and energy into a specific chamber
    void inject(int chamber_idx, Real mass, Real temperature) {
        if (chamber_idx < 0 || chamber_idx >= static_cast<int>(chambers_.size())) return;
        auto& c = chambers_[static_cast<size_t>(chamber_idx)];
        Real cp = c.gamma * c.R_gas / (c.gamma - 1.0);
        Real old_energy = c.mass * cp * c.T;
        Real add_energy = mass * cp * temperature;
        c.mass += mass;
        c.T = (old_energy + add_energy) / (c.mass * cp);
        c.P = c.mass * c.R_gas * c.T / c.V;
    }

    /// Update chamber volume (from structural solver)
    void set_volume(int chamber_idx, Real volume) {
        if (chamber_idx < 0 || chamber_idx >= static_cast<int>(chambers_.size())) return;
        auto& c = chambers_[static_cast<size_t>(chamber_idx)];
        c.V = std::max(volume, 1.0e-10);
        // Isentropic update: P * V^gamma = const (approximate)
        c.P = c.mass * c.R_gas * c.T / c.V;
    }

    size_t num_chambers() const { return chambers_.size(); }
    size_t num_orifices() const { return orifices_.size(); }
    const Chamber& chamber(int i) const { return chambers_[static_cast<size_t>(i)]; }
    Chamber& chamber(int i) { return chambers_[static_cast<size_t>(i)]; }
    Real time() const { return time_; }

private:
    std::vector<Chamber> chambers_;
    std::vector<Orifice> orifices_;
    Real time_ = 0.0;
};

// ============================================================================
// 6. AirbagGasSpecies — Multi-species gas tracking with mixing rules
// ============================================================================

/**
 * @brief Multi-species gas model for airbag inflation.
 *
 * Tracks mass fractions of individual gas species (N2, H2O, CO2, etc.)
 * and computes mixture properties using mass-weighted averaging.
 */
class AirbagGasSpecies {
public:
    struct GasSpecies {
        std::string name;
        Real molar_mass = 28.0e-3; ///< Molar mass [kg/mol]
        Real cp        = 1040.0;   ///< Specific heat at const pressure [J/(kg*K)]
        Real cv        = 743.0;    ///< Specific heat at const volume [J/(kg*K)]
        Real gamma     = 1.4;      ///< cp/cv
    };

    struct MixProperties {
        Real cp_mix     = 0.0;  ///< Mixture cp [J/(kg*K)]
        Real cv_mix     = 0.0;  ///< Mixture cv [J/(kg*K)]
        Real gamma_mix  = 1.4;  ///< Mixture gamma
        Real mw_mix     = 0.0;  ///< Mixture molar mass [kg/mol]
        Real R_mix      = 0.0;  ///< Mixture gas constant [J/(kg*K)]
    };

    AirbagGasSpecies() = default;

    /**
     * @brief Register a gas species.
     * @return Species index
     */
    int add_species(const std::string& name, Real molar_mass, Real cp, Real cv) {
        GasSpecies s;
        s.name = name;
        s.molar_mass = molar_mass;
        s.cp = cp;
        s.cv = cv;
        s.gamma = (cv > 1.0e-15) ? cp / cv : 1.4;
        species_.push_back(s);
        return static_cast<int>(species_.size()) - 1;
    }

    /**
     * @brief Add common inflator gas species (N2, H2O, CO2, Ar, CO).
     */
    void add_common_species() {
        add_species("N2",  28.014e-3, 1040.0, 743.0);
        add_species("H2O", 18.015e-3, 1996.0, 1534.0);
        add_species("CO2", 44.010e-3,  844.0, 655.0);
        add_species("Ar",  39.948e-3,  520.3, 312.2);
        add_species("CO",  28.010e-3, 1040.0, 743.0);
    }

    /**
     * @brief Compute mixture properties from mass fractions.
     *
     * @param mass_fractions Mass fraction of each species (must sum to ~1.0)
     * @return MixProperties with weighted averages
     */
    MixProperties mix(const std::vector<Real>& mass_fractions) const {
        MixProperties result;
        size_t n = std::min(species_.size(), mass_fractions.size());

        Real total_Y = 0.0;
        Real sum_Y_over_M = 0.0;

        for (size_t i = 0; i < n; ++i) {
            Real Y = mass_fractions[i];
            if (Y < 0.0) continue;

            result.cp_mix += Y * species_[i].cp;
            result.cv_mix += Y * species_[i].cv;
            sum_Y_over_M += Y / species_[i].molar_mass;
            total_Y += Y;
        }

        if (total_Y < 1.0e-15) {
            result.gamma_mix = 1.4;
            return result;
        }

        // Normalize
        result.cp_mix /= total_Y;
        result.cv_mix /= total_Y;

        // Mixture molar mass: 1/M_mix = sum(Y_i / M_i)
        if (sum_Y_over_M > 1.0e-30) {
            result.mw_mix = total_Y / sum_Y_over_M;
        }

        // Gamma
        if (result.cv_mix > 1.0e-15) {
            result.gamma_mix = result.cp_mix / result.cv_mix;
        }

        // Gas constant: R = R_universal / M_mix = cp - cv
        result.R_mix = result.cp_mix - result.cv_mix;

        return result;
    }

    /**
     * @brief Compute mixture properties from mole fractions.
     */
    MixProperties mix_molar(const std::vector<Real>& mole_fractions) const {
        size_t n = std::min(species_.size(), mole_fractions.size());

        // Convert mole fractions to mass fractions
        Real total_mass = 0.0;
        std::vector<Real> mass_frac(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            mass_frac[i] = mole_fractions[i] * species_[i].molar_mass;
            total_mass += mass_frac[i];
        }
        if (total_mass > 1.0e-30) {
            for (size_t i = 0; i < n; ++i) {
                mass_frac[i] /= total_mass;
            }
        }
        return mix(mass_frac);
    }

    size_t num_species() const { return species_.size(); }
    const GasSpecies& species(int i) const { return species_[static_cast<size_t>(i)]; }

private:
    std::vector<GasSpecies> species_;
};

// ============================================================================
// 7. AirbagTTF — Tank test format inflator data
// ============================================================================

/**
 * @brief Tank test format (TTF) inflator model.
 *
 * Imports inflator characterization data: mass flow rate vs. time and gas
 * temperature vs. time. Provides interpolation for arbitrary query times.
 */
class AirbagTTF {
public:
    AirbagTTF() = default;

    /**
     * @brief Load TTF data from mass flow and temperature curves.
     *
     * @param mass_flow_time Time points for mass flow [s]
     * @param mass_flow_rate Mass flow rate values [kg/s]
     * @param temp_time Time points for temperature [s]
     * @param temp_values Temperature values [K]
     */
    void load_ttf(const std::vector<Real>& mass_flow_time,
                  const std::vector<Real>& mass_flow_rate,
                  const std::vector<Real>& temp_time,
                  const std::vector<Real>& temp_values)
    {
        mf_time_ = mass_flow_time;
        mf_rate_ = mass_flow_rate;
        t_time_  = temp_time;
        t_vals_  = temp_values;
        loaded_  = true;

        // Compute total mass by integrating mass flow rate
        total_mass_ = 0.0;
        for (size_t i = 1; i < mf_time_.size(); ++i) {
            Real dt = mf_time_[i] - mf_time_[i-1];
            Real avg_rate = 0.5 * (mf_rate_[i-1] + mf_rate_[i]);
            total_mass_ += avg_rate * dt;
        }
    }

    /**
     * @brief Get interpolated mass flow rate at given time.
     */
    Real get_mass_flow(Real time) const {
        if (!loaded_ || mf_time_.empty()) return 0.0;
        return airbag41_detail::interpolate(mf_time_, mf_rate_, time);
    }

    /**
     * @brief Get interpolated gas temperature at given time.
     */
    Real get_temperature(Real time) const {
        if (!loaded_ || t_time_.empty()) return 300.0;
        return airbag41_detail::interpolate(t_time_, t_vals_, time);
    }

    /**
     * @brief Get cumulative injected mass up to given time.
     * Integrates mass flow rate from 0 to t using trapezoidal rule.
     */
    Real get_cumulative_mass(Real time) const {
        if (!loaded_ || mf_time_.empty()) return 0.0;

        Real cumul = 0.0;
        for (size_t i = 1; i < mf_time_.size(); ++i) {
            if (mf_time_[i-1] >= time) break;
            Real t0 = mf_time_[i-1];
            Real t1 = std::min(mf_time_[i], time);
            Real r0 = mf_rate_[i-1];
            Real r1 = airbag41_detail::interpolate(mf_time_, mf_rate_, t1);
            cumul += 0.5 * (r0 + r1) * (t1 - t0);
        }
        return cumul;
    }

    /**
     * @brief Inject into a multi-chamber system at current time.
     */
    void inject_at_time(AirbagMultiChamber& mc, int chamber_idx,
                        Real time, Real dt) const
    {
        Real mdot = get_mass_flow(time);
        Real T    = get_temperature(time);
        Real mass = mdot * dt;
        if (mass > 1.0e-30) {
            mc.inject(chamber_idx, mass, T);
        }
    }

    bool is_loaded() const { return loaded_; }
    Real total_mass() const { return total_mass_; }

    /// Get the time at which mass flow ceases (drops below threshold)
    Real burnout_time(Real threshold = 1.0e-6) const {
        if (!loaded_ || mf_time_.empty()) return 0.0;
        for (size_t i = mf_time_.size(); i > 0; --i) {
            if (mf_rate_[i-1] > threshold) {
                return mf_time_[i-1];
            }
        }
        return 0.0;
    }

private:
    std::vector<Real> mf_time_;  ///< Mass flow time points
    std::vector<Real> mf_rate_;  ///< Mass flow rate values
    std::vector<Real> t_time_;   ///< Temperature time points
    std::vector<Real> t_vals_;   ///< Temperature values
    bool loaded_ = false;
    Real total_mass_ = 0.0;
};

// ============================================================================
// 8. AirbagMembraneDrape — Gravity-driven draping with fold detection
// ============================================================================

/**
 * @brief Membrane draping simulation with fold preservation.
 *
 * Simulates gravity-driven draping of a membrane onto a rigid tool surface.
 * Detects fold regions where the membrane doubles over itself and preserves
 * them during subsequent inflation analysis.
 */
class AirbagMembraneDrape {
public:
    struct DrapeResult {
        std::vector<std::array<Real,3>> deformed_positions; ///< Final node positions
        std::vector<int> fold_regions;   ///< Element indices in fold regions
        Real max_displacement = 0.0;     ///< Maximum node displacement
        int  iterations = 0;             ///< Solver iterations used
        bool converged = false;
    };

    struct ToolSurface {
        /// Tool is defined as a plane z = z_tool (simplified rigid surface)
        Real z_plane = 0.0;
        /// Or as a set of triangles
        std::vector<std::array<Real,9>> triangles; ///< 3 vertices x 3 coords
    };

    AirbagMembraneDrape() = default;

    /**
     * @brief Perform gravity-driven draping of membrane onto tool surface.
     *
     * Uses explicit dynamic relaxation with mass proportional damping.
     *
     * @param nodes Initial node positions (3 per node, flattened)
     * @param elements Element connectivity (3 per triangle, flattened)
     * @param n_nodes Number of nodes
     * @param n_elements Number of elements
     * @param gravity Gravity vector [3]
     * @param tool Tool surface definition
     * @param max_iter Maximum iterations
     * @param tol Convergence tolerance on displacement increment
     * @return DrapeResult with final positions and fold regions
     */
    DrapeResult drape(const std::vector<Real>& nodes,
                      const std::vector<int>& elements,
                      int n_nodes,
                      int n_elements,
                      const Real gravity[3],
                      const ToolSurface& tool,
                      int max_iter = 1000,
                      Real tol = 1.0e-6) const
    {
        DrapeResult result;
        result.deformed_positions.resize(static_cast<size_t>(n_nodes));

        // Initialize positions
        for (int i = 0; i < n_nodes; ++i) {
            size_t idx = static_cast<size_t>(i) * 3;
            if (idx + 2 < nodes.size()) {
                result.deformed_positions[static_cast<size_t>(i)] = {
                    nodes[idx], nodes[idx+1], nodes[idx+2]
                };
            }
        }

        // Velocity array for dynamic relaxation
        std::vector<std::array<Real,3>> vel(static_cast<size_t>(n_nodes), {0.0, 0.0, 0.0});

        Real dt = 1.0e-4;
        Real damping = 0.9;      // Mass-proportional damping coefficient
        Real node_mass = 0.001;  // Lumped mass per node [kg]

        for (int iter = 0; iter < max_iter; ++iter) {
            Real max_disp_inc = 0.0;

            for (int i = 0; i < n_nodes; ++i) {
                auto& pos = result.deformed_positions[static_cast<size_t>(i)];
                auto& v   = vel[static_cast<size_t>(i)];

                // Gravity force
                Real fx = node_mass * gravity[0];
                Real fy = node_mass * gravity[1];
                Real fz = node_mass * gravity[2];

                // Damped explicit update: a = (F - c*m*v) / m
                Real ax = fx / node_mass - damping * v[0];
                Real ay = fy / node_mass - damping * v[1];
                Real az = fz / node_mass - damping * v[2];

                v[0] += ax * dt;
                v[1] += ay * dt;
                v[2] += az * dt;

                Real dx = v[0] * dt;
                Real dy = v[1] * dt;
                Real dz = v[2] * dt;

                pos[0] += dx;
                pos[1] += dy;
                pos[2] += dz;

                // Contact with tool surface (simple plane contact)
                if (pos[2] < tool.z_plane) {
                    pos[2] = tool.z_plane;
                    if (v[2] < 0.0) v[2] = 0.0;
                }

                // Contact with tool triangles
                for (const auto& tri : tool.triangles) {
                    // Simple point-in-triangle + z-projection contact
                    Real z_tool = std::min({tri[2], tri[5], tri[8]});
                    if (pos[2] < z_tool) {
                        pos[2] = z_tool;
                        if (v[2] < 0.0) v[2] = 0.0;
                    }
                }

                Real disp_inc = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (disp_inc > max_disp_inc) max_disp_inc = disp_inc;
            }

            result.iterations = iter + 1;
            if (max_disp_inc < tol) {
                result.converged = true;
                break;
            }
        }

        // Compute max total displacement
        for (int i = 0; i < n_nodes; ++i) {
            size_t idx = static_cast<size_t>(i) * 3;
            if (idx + 2 >= nodes.size()) continue;
            auto& pos = result.deformed_positions[static_cast<size_t>(i)];
            Real dx = pos[0] - nodes[idx];
            Real dy = pos[1] - nodes[idx+1];
            Real dz = pos[2] - nodes[idx+2];
            Real d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d > result.max_displacement) result.max_displacement = d;
        }

        return result;
    }

    /**
     * @brief Detect fold regions where membrane folds over itself.
     *
     * A fold is detected when adjacent element normals differ by more than
     * the threshold angle (in radians).
     *
     * @param positions Node positions (array of [x,y,z])
     * @param elements Element connectivity (3 per triangle, flattened)
     * @param n_elements Number of elements
     * @param threshold_angle Angle threshold in radians (default pi/4)
     * @return Indices of elements in fold regions
     */
    std::vector<int> detect_folds(
        const std::vector<std::array<Real,3>>& positions,
        const std::vector<int>& elements,
        int n_elements,
        Real threshold_angle = M_PI / 4.0) const
    {
        // Compute element normals
        std::vector<std::array<Real,3>> normals(static_cast<size_t>(n_elements));

        for (int e = 0; e < n_elements; ++e) {
            size_t base = static_cast<size_t>(e) * 3;
            if (base + 2 >= elements.size()) continue;

            int n0 = elements[base];
            int n1 = elements[base + 1];
            int n2 = elements[base + 2];

            if (n0 < 0 || n1 < 0 || n2 < 0) continue;
            size_t sn0 = static_cast<size_t>(n0);
            size_t sn1 = static_cast<size_t>(n1);
            size_t sn2 = static_cast<size_t>(n2);
            if (sn0 >= positions.size() || sn1 >= positions.size() ||
                sn2 >= positions.size()) continue;

            const auto& p0 = positions[sn0];
            const auto& p1 = positions[sn1];
            const auto& p2 = positions[sn2];

            Real e1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            Real e2[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};

            // Cross product for normal
            normals[static_cast<size_t>(e)] = {
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0]
            };

            Real len = airbag41_detail::norm3(normals[static_cast<size_t>(e)].data());
            if (len > 1.0e-30) {
                normals[static_cast<size_t>(e)][0] /= len;
                normals[static_cast<size_t>(e)][1] /= len;
                normals[static_cast<size_t>(e)][2] /= len;
            }
        }

        // Find folds: compare adjacent element normals
        // Two elements are adjacent if they share an edge (2 nodes)
        std::vector<int> fold_elements;
        Real cos_threshold = std::cos(threshold_angle);

        for (int e1 = 0; e1 < n_elements; ++e1) {
            size_t base1 = static_cast<size_t>(e1) * 3;
            if (base1 + 2 >= elements.size()) continue;

            for (int e2 = e1 + 1; e2 < n_elements; ++e2) {
                size_t base2 = static_cast<size_t>(e2) * 3;
                if (base2 + 2 >= elements.size()) continue;

                // Check shared nodes (adjacency)
                int shared = 0;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (elements[base1 + static_cast<size_t>(i)] ==
                            elements[base2 + static_cast<size_t>(j)]) {
                            ++shared;
                        }
                    }
                }

                if (shared >= 2) {
                    // Adjacent elements: compare normals
                    Real dot = airbag41_detail::dot3(
                        normals[static_cast<size_t>(e1)].data(),
                        normals[static_cast<size_t>(e2)].data());

                    if (dot < cos_threshold) {
                        // Fold detected
                        fold_elements.push_back(e1);
                        fold_elements.push_back(e2);
                    }
                }
            }
        }

        // Remove duplicates
        std::sort(fold_elements.begin(), fold_elements.end());
        fold_elements.erase(
            std::unique(fold_elements.begin(), fold_elements.end()),
            fold_elements.end());

        return fold_elements;
    }
};

} // namespace fem
} // namespace nxs
