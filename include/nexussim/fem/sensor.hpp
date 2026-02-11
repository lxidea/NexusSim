#pragma once

/**
 * @file sensor.hpp
 * @brief Virtual sensor system for simulation monitoring
 *
 * Sensors:
 * - Accelerometer: measures acceleration at a node (with optional filtering)
 * - Velocity gauge: measures velocity at a node
 * - Strain gauge: measures strain at an element (rosette or uniaxial)
 * - Force sensor: measures reaction force at a node
 * - Distance sensor: measures distance between two nodes
 *
 * Features:
 * - Configurable sampling rate (independent of time step)
 * - CFC (Channel Frequency Class) filtering for accelerometers
 * - Threshold detection for triggering controls
 * - Time history output
 *
 * Reference: LS-DYNA *DATABASE_HISTORY_NODE, *ELEMENT_SEATBELT_ACCELEROMETER
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nxs {
namespace fem {

// ============================================================================
// Sensor Types
// ============================================================================

enum class SensorType {
    Accelerometer,    ///< Acceleration at a node
    VelocityGauge,    ///< Velocity at a node
    StrainGauge,      ///< Strain at an element
    ForceSensor,      ///< Force at a node
    DistanceSensor    ///< Distance between two nodes
};

enum class SensorDirection {
    X, Y, Z,
    Magnitude,        ///< |v| = sqrt(vx²+vy²+vz²)
    ResultantXY       ///< sqrt(vx²+vy²) — typical for crash
};

// ============================================================================
// CFC Filter (SAE J211 - Butterworth low-pass)
// ============================================================================

/**
 * @brief Simple 2nd-order Butterworth low-pass filter
 *
 * Used for CFC60, CFC180, CFC600, CFC1000 filtering
 * of accelerometer data per SAE J211 standard.
 */
struct CFCFilter {
    Real cutoff_freq;   ///< Cutoff frequency (Hz)
    bool enabled;

    // Filter state (2nd order IIR)
    Real a1, a2, b0, b1, b2;
    Real x_prev[2];    ///< Previous input values
    Real y_prev[2];    ///< Previous output values

    CFCFilter() : cutoff_freq(0.0), enabled(false)
        , a1(0), a2(0), b0(1), b1(0), b2(0) {
        x_prev[0] = x_prev[1] = 0.0;
        y_prev[0] = y_prev[1] = 0.0;
    }

    /**
     * @brief Initialize Butterworth filter coefficients
     * @param cfc CFC class (60, 180, 600, 1000)
     * @param dt Sampling time step
     */
    void setup(int cfc, Real dt) {
        enabled = true;
        // CFC frequency to cutoff: f_cutoff = CFC * 1.0 Hz (per SAE J211)
        cutoff_freq = static_cast<Real>(cfc);

        Real omega = 2.0 * constants::pi<Real> * cutoff_freq;
        Real T = dt;
        Real c = omega * T / 2.0;

        // Bilinear transform of 2nd-order Butterworth
        Real k = std::tan(c);
        Real k2 = k * k;
        Real sqrt2 = constants::sqrt_two<Real>;
        Real denom = 1.0 + sqrt2 * k + k2;

        b0 = k2 / denom;
        b1 = 2.0 * k2 / denom;
        b2 = k2 / denom;
        a1 = 2.0 * (k2 - 1.0) / denom;
        a2 = (1.0 - sqrt2 * k + k2) / denom;

        x_prev[0] = x_prev[1] = 0.0;
        y_prev[0] = y_prev[1] = 0.0;
    }

    /**
     * @brief Apply filter to one sample
     */
    Real apply(Real x) {
        if (!enabled) return x;

        Real y = b0 * x + b1 * x_prev[0] + b2 * x_prev[1]
                 - a1 * y_prev[0] - a2 * y_prev[1];

        x_prev[1] = x_prev[0];
        x_prev[0] = x;
        y_prev[1] = y_prev[0];
        y_prev[0] = y;

        return y;
    }

    void reset() {
        x_prev[0] = x_prev[1] = 0.0;
        y_prev[0] = y_prev[1] = 0.0;
    }
};

// ============================================================================
// Sensor Configuration
// ============================================================================

struct SensorConfig {
    SensorType type;
    int id;
    std::string name;

    // Location
    Index node_id;           ///< Primary node (all sensor types)
    Index node_id2;          ///< Secondary node (distance sensor)
    Index element_id;        ///< Element (strain gauge)

    // Measurement
    SensorDirection direction;
    int cfc_class;           ///< CFC filter class (0=none, 60/180/600/1000)

    // Threshold
    Real threshold_value;    ///< Trigger threshold
    bool threshold_above;    ///< true=trigger when above, false=when below

    // Sampling
    Real sample_interval;    ///< Minimum time between samples (0=every step)

    SensorConfig()
        : type(SensorType::Accelerometer), id(0)
        , node_id(0), node_id2(0), element_id(0)
        , direction(SensorDirection::Magnitude)
        , cfc_class(0)
        , threshold_value(1.0e30), threshold_above(true)
        , sample_interval(0.0) {}
};

// ============================================================================
// Sensor Reading
// ============================================================================

struct SensorReading {
    Real time;
    Real raw_value;       ///< Unfiltered value
    Real filtered_value;  ///< CFC-filtered value
    bool threshold_exceeded;
};

// ============================================================================
// Sensor
// ============================================================================

class Sensor {
public:
    explicit Sensor(const SensorConfig& cfg)
        : config_(cfg), last_sample_time_(-1.0e30)
        , threshold_triggered_(false) {
        if (cfg.cfc_class > 0) {
            filter_.setup(cfg.cfc_class, 1.0e-5);  // Default dt, updated on first measure
        }
    }

    const SensorConfig& config() const { return config_; }
    int id() const { return config_.id; }
    const std::string& name() const { return config_.name; }
    bool threshold_triggered() const { return threshold_triggered_; }
    Real current_value() const { return readings_.empty() ? 0.0 : readings_.back().filtered_value; }

    /**
     * @brief Take a measurement
     *
     * @param time Current simulation time
     * @param dt Current time step (for filter update)
     * @param num_nodes Total number of nodes
     * @param positions Node positions (3*num_nodes)
     * @param velocities Node velocities (3*num_nodes)
     * @param accelerations Node accelerations (3*num_nodes)
     * @param forces Node forces (3*num_nodes)
     * @param num_elements Total number of elements
     * @param strains Element strains (6*num_elements)
     */
    void measure(Real time, Real dt,
                 std::size_t num_nodes,
                 const Real* positions = nullptr,
                 const Real* velocities = nullptr,
                 const Real* accelerations = nullptr,
                 const Real* forces = nullptr,
                 std::size_t num_elements = 0,
                 const Real* strains = nullptr) {

        // Check sampling interval
        if (config_.sample_interval > 0.0 &&
            (time - last_sample_time_) < config_.sample_interval * 0.999) {
            return;
        }
        last_sample_time_ = time;

        // Update filter dt if needed
        if (config_.cfc_class > 0 && dt > 0.0 && !filter_dt_set_) {
            filter_.setup(config_.cfc_class, dt);
            filter_dt_set_ = true;
        }

        // Extract raw value
        Real raw = extract_value(num_nodes, positions, velocities,
                                  accelerations, forces,
                                  num_elements, strains);

        // Apply filter
        Real filtered = (config_.cfc_class > 0) ? filter_.apply(raw) : raw;

        // Check threshold
        bool exceeded = config_.threshold_above
            ? (filtered > config_.threshold_value)
            : (filtered < config_.threshold_value);
        if (exceeded) threshold_triggered_ = true;

        readings_.push_back({time, raw, filtered, exceeded});
    }

    // --- Data Access ---

    const std::vector<SensorReading>& readings() const { return readings_; }
    std::size_t num_readings() const { return readings_.size(); }

    void clear_readings() {
        readings_.clear();
        filter_.reset();
        threshold_triggered_ = false;
    }

    void print_summary() const {
        std::cout << "Sensor [" << config_.id << "] " << config_.name
                  << ": " << readings_.size() << " readings";
        if (!readings_.empty()) {
            std::cout << ", latest=" << readings_.back().filtered_value;
        }
        if (threshold_triggered_) std::cout << " [THRESHOLD]";
        std::cout << "\n";
    }

private:
    Real extract_value(std::size_t num_nodes,
                        const Real* pos, const Real* vel,
                        const Real* accel, const Real* force,
                        std::size_t num_elements,
                        const Real* strains) const {

        switch (config_.type) {
            case SensorType::Accelerometer:
                return extract_nodal(config_.node_id, accel, num_nodes);
            case SensorType::VelocityGauge:
                return extract_nodal(config_.node_id, vel, num_nodes);
            case SensorType::ForceSensor:
                return extract_nodal(config_.node_id, force, num_nodes);
            case SensorType::DistanceSensor:
                return extract_distance(pos, num_nodes);
            case SensorType::StrainGauge:
                return extract_strain(strains, num_elements);
            default:
                return 0.0;
        }
    }

    Real extract_nodal(Index nid, const Real* data, std::size_t num_nodes) const {
        if (!data || nid >= num_nodes) return 0.0;
        Real vx = data[3*nid+0];
        Real vy = data[3*nid+1];
        Real vz = data[3*nid+2];

        switch (config_.direction) {
            case SensorDirection::X: return vx;
            case SensorDirection::Y: return vy;
            case SensorDirection::Z: return vz;
            case SensorDirection::Magnitude:
                return std::sqrt(vx*vx + vy*vy + vz*vz);
            case SensorDirection::ResultantXY:
                return std::sqrt(vx*vx + vy*vy);
            default: return 0.0;
        }
    }

    Real extract_distance(const Real* pos, std::size_t num_nodes) const {
        if (!pos || config_.node_id >= num_nodes || config_.node_id2 >= num_nodes) return 0.0;
        Real dx = pos[3*config_.node_id+0] - pos[3*config_.node_id2+0];
        Real dy = pos[3*config_.node_id+1] - pos[3*config_.node_id2+1];
        Real dz = pos[3*config_.node_id+2] - pos[3*config_.node_id2+2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    Real extract_strain(const Real* strains, std::size_t num_elements) const {
        if (!strains || config_.element_id >= num_elements) return 0.0;
        Index eid = config_.element_id;
        // Return effective strain or component based on direction
        switch (config_.direction) {
            case SensorDirection::X: return strains[6*eid+0];
            case SensorDirection::Y: return strains[6*eid+1];
            case SensorDirection::Z: return strains[6*eid+2];
            case SensorDirection::Magnitude: {
                // Effective strain (von Mises equivalent)
                Real e1=strains[6*eid+0], e2=strains[6*eid+1], e3=strains[6*eid+2];
                Real e4=strains[6*eid+3], e5=strains[6*eid+4], e6=strains[6*eid+5];
                Real d1=e1-e2, d2=e2-e3, d3=e1-e3;
                return std::sqrt(2.0/9.0*(d1*d1+d2*d2+d3*d3) +
                                 (e4*e4+e5*e5+e6*e6)/3.0);
            }
            default: return 0.0;
        }
    }

    SensorConfig config_;
    CFCFilter filter_;
    bool filter_dt_set_ = false;
    Real last_sample_time_;
    bool threshold_triggered_;
    std::vector<SensorReading> readings_;
};

// ============================================================================
// Sensor Manager
// ============================================================================

class SensorManager {
public:
    SensorManager() = default;

    Sensor& add_sensor(const SensorConfig& cfg) {
        sensors_.emplace_back(cfg);
        return sensors_.back();
    }

    void measure_all(Real time, Real dt, std::size_t num_nodes,
                     const Real* pos = nullptr,
                     const Real* vel = nullptr,
                     const Real* accel = nullptr,
                     const Real* force = nullptr,
                     std::size_t num_elements = 0,
                     const Real* strains = nullptr) {
        for (auto& s : sensors_) {
            s.measure(time, dt, num_nodes, pos, vel, accel, force,
                      num_elements, strains);
        }
    }

    Sensor* find(int id) {
        for (auto& s : sensors_) {
            if (s.id() == id) return &s;
        }
        return nullptr;
    }

    const Sensor* find(int id) const {
        for (const auto& s : sensors_) {
            if (s.id() == id) return &s;
        }
        return nullptr;
    }

    bool any_threshold_triggered() const {
        for (const auto& s : sensors_) {
            if (s.threshold_triggered()) return true;
        }
        return false;
    }

    std::vector<int> triggered_sensor_ids() const {
        std::vector<int> ids;
        for (const auto& s : sensors_) {
            if (s.threshold_triggered()) ids.push_back(s.id());
        }
        return ids;
    }

    std::size_t num_sensors() const { return sensors_.size(); }

    void print_summary() const {
        std::cout << "Sensors: " << sensors_.size() << "\n";
        for (const auto& s : sensors_) s.print_summary();
    }

private:
    std::vector<Sensor> sensors_;
};

} // namespace fem
} // namespace nxs
