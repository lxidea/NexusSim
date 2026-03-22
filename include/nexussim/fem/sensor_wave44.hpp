#pragma once

/**
 * @file sensor_wave44.hpp
 * @brief Sensor aggregation and expression system for simulation monitoring
 *
 * Wave 44 additions:
 * - SensorAggregator: windowed aggregation (Min/Max/Mean/RMS/Envelope)
 * - SensorExpression: postfix expression evaluator over sensor values
 * - SensorExpressionTrigger: callback-based trigger from sensor expressions
 * - MultiSensorAggregator: manage aggregators for multiple sensors at once
 *
 * Reference: OpenRadioss /SENSOR, /COND, /MONVOL threshold logic
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/sensor.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nxs {
namespace fem {

// ============================================================================
// AggregationType
// ============================================================================

/**
 * @brief Aggregation method for a time-windowed collection of sensor samples.
 */
enum class AggregationType {
    Min,        ///< Minimum value in the window
    Max,        ///< Maximum value in the window
    Mean,       ///< Arithmetic mean of values in the window
    RMS,        ///< Root-mean-square of values in the window
    Envelope    ///< Maximum absolute value in the window
};

// ============================================================================
// SensorAggregator
// ============================================================================

/**
 * @brief Collects sensor samples over a sliding time window and computes
 *        aggregate statistics.
 *
 * Usage:
 * @code
 *   SensorAggregator agg;
 *   agg.configure(0.01, AggregationType::RMS);   // 10 ms window
 *   agg.push(t, value);
 *   Real rms = agg.result();
 * @endcode
 */
class SensorAggregator {
public:
    SensorAggregator() = default;

    /**
     * @brief Configure the aggregator.
     * @param window_size  Length of the sliding time window (seconds).
     *                     Use 0.0 or negative to keep all samples.
     * @param type         Aggregation method.
     */
    void configure(Real window_size, AggregationType type) {
        window_size_ = window_size;
        type_        = type;
    }

    /**
     * @brief Add a sample and trim stale entries outside the window.
     * @param time  Simulation time of the sample.
     * @param value Measured value.
     */
    void push(Real time, Real value) {
        samples_.emplace_back(time, value);
        trim(time);
    }

    /**
     * @brief Return the aggregated result over all samples in the window.
     * @return 0.0 if no samples are present.
     */
    Real result() const {
        if (samples_.empty()) return Real(0);

        switch (type_) {
            case AggregationType::Min: {
                Real v = samples_[0].second;
                for (const auto& s : samples_) v = std::min(v, s.second);
                return v;
            }
            case AggregationType::Max: {
                Real v = samples_[0].second;
                for (const auto& s : samples_) v = std::max(v, s.second);
                return v;
            }
            case AggregationType::Mean: {
                Real sum = Real(0);
                for (const auto& s : samples_) sum += s.second;
                return sum / static_cast<Real>(samples_.size());
            }
            case AggregationType::RMS: {
                Real sum2 = Real(0);
                for (const auto& s : samples_) sum2 += s.second * s.second;
                return std::sqrt(sum2 / static_cast<Real>(samples_.size()));
            }
            case AggregationType::Envelope: {
                Real v = Real(0);
                for (const auto& s : samples_) v = std::max(v, std::abs(s.second));
                return v;
            }
            default:
                return Real(0);
        }
    }

    /** @brief Remove all stored samples and reset the aggregator. */
    void reset() { samples_.clear(); }

    /** @brief Number of samples currently in the window. */
    std::size_t num_samples() const { return samples_.size(); }

    /** @brief Configured time window size (0 = unlimited). */
    Real window_size() const { return window_size_; }

    /** @brief Configured aggregation type. */
    AggregationType type() const { return type_; }

private:
    /** Discard samples whose time is older than (current_time - window_size_). */
    void trim(Real current_time) {
        if (window_size_ <= Real(0)) return;  // unlimited window
        Real cutoff = current_time - window_size_;
        auto it = std::remove_if(samples_.begin(), samples_.end(),
            [cutoff](const std::pair<Real, Real>& s) {
                return s.first < cutoff;
            });
        samples_.erase(it, samples_.end());
    }

    Real window_size_                = Real(0);
    AggregationType type_            = AggregationType::Mean;
    std::vector<std::pair<Real, Real>> samples_;  ///< (time, value)
};

// ============================================================================
// Expression token types
// ============================================================================

/** @brief Token categories for the postfix sensor expression language. */
enum class ExprTokenType {
    Sensor,       ///< Sensor value: operand identified by sensor_id
    Aggregation,  ///< Aggregation result operand identified by agg_type
    Constant,     ///< Numeric literal operand
    Add,          ///< Binary +
    Sub,          ///< Binary -
    Mul,          ///< Binary *
    Div,          ///< Binary /
    GreaterThan,  ///< Binary >  (returns 1.0 / 0.0)
    LessThan,     ///< Binary <  (returns 1.0 / 0.0)
    And,          ///< Logical AND (both operands > 0.5)
    Or,           ///< Logical OR  (either operand > 0.5)
    Negate        ///< Unary negation (~ token)
};

/** @brief A single token in a postfix sensor expression. */
struct ExprToken {
    ExprTokenType type             = ExprTokenType::Constant;
    int           sensor_id        = 0;
    AggregationType agg_type       = AggregationType::Mean;
    Real          constant_value   = Real(0);

    // --- Factory helpers ---

    static ExprToken make_sensor(int id) {
        ExprToken t;
        t.type      = ExprTokenType::Sensor;
        t.sensor_id = id;
        return t;
    }
    static ExprToken make_constant(Real v) {
        ExprToken t;
        t.type           = ExprTokenType::Constant;
        t.constant_value = v;
        return t;
    }
    static ExprToken make_aggregation(AggregationType a) {
        ExprToken t;
        t.type     = ExprTokenType::Aggregation;
        t.agg_type = a;
        return t;
    }
    static ExprToken make_op(ExprTokenType op) {
        ExprToken t;
        t.type = op;
        return t;
    }
};

// ============================================================================
// SensorExpression
// ============================================================================

/**
 * @brief Postfix (reverse-polish) expression evaluator for sensor values.
 *
 * The expression is stored as a flat vector of ExprToken objects in postfix
 * order.  Evaluation uses a simple value stack.
 *
 * Supported parse format (space-separated tokens):
 *   S<id>          → Sensor(id)          e.g. "S1"
 *   C<value>       → Constant            e.g. "C3.14"
 *   AGG_MIN/MAX/MEAN/RMS/ENV → Aggregation token
 *   + - * /        → arithmetic operators
 *   > <            → comparison operators (result 1.0 or 0.0)
 *   & |            → logical AND / OR
 *   ~              → unary negate
 *
 * Example: "S1 C100.0 >"  evaluates to 1.0 when sensor 1 > 100.0
 */
class SensorExpression {
public:
    /**
     * @brief Construct from a pre-built postfix token sequence.
     * @param tokens  Postfix-ordered token list.
     */
    explicit SensorExpression(std::vector<ExprToken> tokens)
        : tokens_(std::move(tokens)) {}

    /**
     * @brief Evaluate the expression against a SensorManager.
     *
     * Sensor tokens read the sensor's current_value().
     * Missing sensors return 0.0.
     *
     * @throws std::runtime_error if the token sequence is malformed.
     */
    Real evaluate(const SensorManager& mgr) const {
        std::vector<Real> stack;
        stack.reserve(tokens_.size());

        for (const auto& tok : tokens_) {
            switch (tok.type) {
                case ExprTokenType::Sensor: {
                    const Sensor* s = mgr.find(tok.sensor_id);
                    stack.push_back(s ? s->current_value() : Real(0));
                    break;
                }
                case ExprTokenType::Constant:
                    stack.push_back(tok.constant_value);
                    break;
                case ExprTokenType::Aggregation:
                    // Aggregation tokens push 0 here; MultiSensorAggregator
                    // must supply the value externally when needed.
                    stack.push_back(Real(0));
                    break;
                case ExprTokenType::Negate: {
                    if (stack.empty())
                        throw std::runtime_error("SensorExpression: stack underflow (negate)");
                    stack.back() = -stack.back();
                    break;
                }
                // Binary operators
                case ExprTokenType::Add:
                case ExprTokenType::Sub:
                case ExprTokenType::Mul:
                case ExprTokenType::Div:
                case ExprTokenType::GreaterThan:
                case ExprTokenType::LessThan:
                case ExprTokenType::And:
                case ExprTokenType::Or: {
                    if (stack.size() < 2)
                        throw std::runtime_error("SensorExpression: stack underflow (binary op)");
                    Real b = stack.back(); stack.pop_back();
                    Real a = stack.back(); stack.pop_back();
                    stack.push_back(apply_binary(tok.type, a, b));
                    break;
                }
                default:
                    throw std::runtime_error("SensorExpression: unknown token type");
            }
        }

        if (stack.size() != 1)
            throw std::runtime_error("SensorExpression: expression did not reduce to single value");
        return stack[0];
    }

    /**
     * @brief Parse a space-separated postfix expression string.
     * @param expr  Expression string (see class documentation for format).
     * @return SensorExpression ready for evaluation.
     */
    static SensorExpression parse(const std::string& expr) {
        std::vector<ExprToken> tokens;
        std::istringstream iss(expr);
        std::string tok;
        while (iss >> tok) {
            if (tok.size() >= 2 && tok[0] == 'S') {
                // Sensor token: S<id>
                int id = std::stoi(tok.substr(1));
                tokens.push_back(ExprToken::make_sensor(id));
            } else if (tok.size() >= 2 && tok[0] == 'C') {
                // Constant token: C<value>
                Real v = static_cast<Real>(std::stod(tok.substr(1)));
                tokens.push_back(ExprToken::make_constant(v));
            } else if (tok == "AGG_MIN") {
                tokens.push_back(ExprToken::make_aggregation(AggregationType::Min));
            } else if (tok == "AGG_MAX") {
                tokens.push_back(ExprToken::make_aggregation(AggregationType::Max));
            } else if (tok == "AGG_MEAN") {
                tokens.push_back(ExprToken::make_aggregation(AggregationType::Mean));
            } else if (tok == "AGG_RMS") {
                tokens.push_back(ExprToken::make_aggregation(AggregationType::RMS));
            } else if (tok == "AGG_ENV") {
                tokens.push_back(ExprToken::make_aggregation(AggregationType::Envelope));
            } else if (tok == "+") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Add));
            } else if (tok == "-") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Sub));
            } else if (tok == "*") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Mul));
            } else if (tok == "/") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Div));
            } else if (tok == ">") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::GreaterThan));
            } else if (tok == "<") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::LessThan));
            } else if (tok == "&") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::And));
            } else if (tok == "|") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Or));
            } else if (tok == "~") {
                tokens.push_back(ExprToken::make_op(ExprTokenType::Negate));
            } else {
                // Attempt to parse as bare numeric constant
                try {
                    Real v = static_cast<Real>(std::stod(tok));
                    tokens.push_back(ExprToken::make_constant(v));
                } catch (...) {
                    throw std::runtime_error(
                        std::string("SensorExpression::parse: unrecognised token '") + tok + "'");
                }
            }
        }
        return SensorExpression(std::move(tokens));
    }

    /** @brief Number of tokens in the expression. */
    std::size_t num_tokens() const { return tokens_.size(); }

    /**
     * @brief Basic validation: non-empty and stack simulation ends with depth 1.
     * @return true if the expression appears well-formed.
     */
    bool is_valid() const {
        if (tokens_.empty()) return false;
        int depth = 0;
        for (const auto& tok : tokens_) {
            switch (tok.type) {
                case ExprTokenType::Sensor:
                case ExprTokenType::Constant:
                case ExprTokenType::Aggregation:
                    ++depth;
                    break;
                case ExprTokenType::Negate:
                    if (depth < 1) return false;
                    // depth unchanged
                    break;
                case ExprTokenType::Add:
                case ExprTokenType::Sub:
                case ExprTokenType::Mul:
                case ExprTokenType::Div:
                case ExprTokenType::GreaterThan:
                case ExprTokenType::LessThan:
                case ExprTokenType::And:
                case ExprTokenType::Or:
                    if (depth < 2) return false;
                    --depth;
                    break;
                default:
                    return false;
            }
        }
        return depth == 1;
    }

    /** @brief Read-only access to the token list. */
    const std::vector<ExprToken>& tokens() const { return tokens_; }

private:
    static Real apply_binary(ExprTokenType op, Real a, Real b) {
        switch (op) {
            case ExprTokenType::Add:         return a + b;
            case ExprTokenType::Sub:         return a - b;
            case ExprTokenType::Mul:         return a * b;
            case ExprTokenType::Div:         return (b != Real(0)) ? a / b : Real(0);
            case ExprTokenType::GreaterThan: return (a > b) ? Real(1) : Real(0);
            case ExprTokenType::LessThan:    return (a < b) ? Real(1) : Real(0);
            case ExprTokenType::And:         return (a > Real(0.5) && b > Real(0.5)) ? Real(1) : Real(0);
            case ExprTokenType::Or:          return (a > Real(0.5) || b > Real(0.5)) ? Real(1) : Real(0);
            default:                         return Real(0);
        }
    }

    std::vector<ExprToken> tokens_;
};

// ============================================================================
// SensorExpressionTrigger
// ============================================================================

/**
 * @brief Connects a SensorExpression to an output action via a threshold.
 *
 * When the expression result crosses the threshold (above or below),
 * an optional user callback is fired.
 *
 * Usage:
 * @code
 *   auto expr = SensorExpression::parse("S1 C100.0 >");
 *   SensorExpressionTrigger trigger(expr, 0.5, true);  // fire when result > 0.5
 *   trigger.set_callback([](){ std::cout << "triggered!\n"; });
 *   trigger.evaluate_and_trigger(mgr);
 * @endcode
 */
class SensorExpressionTrigger {
public:
    /**
     * @brief Construct a trigger.
     * @param expr        Expression to evaluate.
     * @param threshold   Comparison value.
     * @param above       If true, trigger when expression > threshold;
     *                    if false, trigger when expression < threshold.
     */
    SensorExpressionTrigger(SensorExpression expr, Real threshold, bool above = true)
        : expr_(std::move(expr))
        , threshold_(threshold)
        , above_(above) {}

    /**
     * @brief Evaluate the expression and return true if the trigger fires.
     * @param mgr  SensorManager holding current sensor values.
     */
    bool check(const SensorManager& mgr) const {
        Real val = expr_.evaluate(mgr);
        return above_ ? (val > threshold_) : (val < threshold_);
    }

    /**
     * @brief Set the callback invoked when the trigger fires.
     * @param cb  Callable taking no arguments.
     */
    void set_callback(std::function<void()> cb) {
        callback_ = std::move(cb);
    }

    /**
     * @brief Evaluate the expression; if triggered, invoke the callback.
     * @param mgr  SensorManager holding current sensor values.
     */
    void evaluate_and_trigger(const SensorManager& mgr) {
        if (check(mgr) && callback_) {
            callback_();
        }
    }

    /** @brief Read-only access to the underlying expression. */
    const SensorExpression& expression() const { return expr_; }

    /** @brief Threshold value. */
    Real threshold() const { return threshold_; }

    /** @brief True if triggering on "above threshold". */
    bool trigger_above() const { return above_; }

private:
    SensorExpression        expr_;
    Real                    threshold_;
    bool                    above_;
    std::function<void()>   callback_;
};

// ============================================================================
// MultiSensorAggregator
// ============================================================================

/**
 * @brief Manages a collection of SensorAggregator instances keyed by sensor ID.
 *
 * Useful for batch operations: configure all sensors with the same window and
 * aggregation type, then push all values at once each time step.
 *
 * Usage:
 * @code
 *   MultiSensorAggregator msa;
 *   msa.configure_all({1, 2, 3}, 0.05, AggregationType::Max);
 *   msa.push_all(mgr, current_time);
 *   Real peak_s2 = msa.get_result(2);
 * @endcode
 */
class MultiSensorAggregator {
public:
    MultiSensorAggregator() = default;

    /**
     * @brief Configure (or re-configure) aggregators for the given sensor IDs.
     *
     * Existing aggregators for listed IDs are replaced; aggregators for IDs
     * not in the list are left unchanged.
     *
     * @param sensor_ids  Sensor IDs to configure.
     * @param window      Time window size (seconds).
     * @param type        Aggregation method.
     */
    void configure_all(const std::vector<int>& sensor_ids,
                       Real window,
                       AggregationType type) {
        for (int id : sensor_ids) {
            aggregators_[id].configure(window, type);
        }
    }

    /**
     * @brief Push the current value of every configured sensor.
     *
     * Sensors that do not exist in @p mgr contribute 0.0.
     *
     * @param mgr   SensorManager to read from.
     * @param time  Current simulation time.
     */
    void push_all(const SensorManager& mgr, Real time) {
        for (auto& [id, agg] : aggregators_) {
            const Sensor* s = mgr.find(id);
            Real val = s ? s->current_value() : Real(0);
            agg.push(time, val);
        }
    }

    /**
     * @brief Return the aggregated result for one sensor.
     * @param sensor_id  Sensor ID (must have been configured).
     * @return Aggregated result, or 0.0 if the ID is unknown.
     */
    Real get_result(int sensor_id) const {
        auto it = aggregators_.find(sensor_id);
        if (it == aggregators_.end()) return Real(0);
        return it->second.result();
    }

    /**
     * @brief Access the underlying SensorAggregator for one sensor (read-only).
     * @param sensor_id  Sensor ID.
     * @return Pointer to aggregator, or nullptr if not configured.
     */
    const SensorAggregator* get_aggregator(int sensor_id) const {
        auto it = aggregators_.find(sensor_id);
        return (it != aggregators_.end()) ? &it->second : nullptr;
    }

    /** @brief Number of configured aggregators. */
    std::size_t num_aggregators() const { return aggregators_.size(); }

    /** @brief Reset all aggregators (clear samples). */
    void reset_all() {
        for (auto& [id, agg] : aggregators_) agg.reset();
    }

private:
    std::unordered_map<int, SensorAggregator> aggregators_;
};

} // namespace fem
} // namespace nxs
