#pragma once

/**
 * @file config_reader.hpp
 * @brief Configuration file reader for simulation parameters
 *
 * Supports simple YAML-like format:
 * ```
 * simulation:
 *   name: "my_simulation"
 *   time_step: 1.0e-4
 *   final_time: 0.01
 *
 * mesh:
 *   file: "mesh.mesh"
 *
 * materials:
 *   - name: "steel"
 *     density: 7850.0
 *     E: 200.0e9
 *     nu: 0.3
 *
 * boundary_conditions:
 *   - type: "displacement"
 *     nodeset: "fixed"
 *     dof: [0, 1, 2]
 *     value: 0.0
 *   - type: "force"
 *     nodeset: "loaded"
 *     dof: 2
 *     value: -1000.0
 *
 * output:
 *   format: "vtk"
 *   frequency: 10
 *   directory: "results"
 * ```
 */

#include <nexussim/core/core.hpp>
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <memory>

namespace nxs {
namespace io {

/**
 * @brief Configuration value types
 */
using ConfigValue = std::variant<
    std::string,
    Real,
    Int,
    bool,
    std::vector<Real>,
    std::vector<Int>,
    std::vector<std::string>
>;

/**
 * @brief Configuration section (hierarchical key-value storage)
 */
class ConfigSection {
public:
    ConfigSection() = default;
    explicit ConfigSection(const std::string& name) : name_(name) {}

    /**
     * @brief Get section name
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Check if key exists
     */
    bool has(const std::string& key) const {
        return values_.count(key) > 0;
    }

    /**
     * @brief Get value as string
     */
    std::string get_string(const std::string& key, const std::string& default_val = "") const;

    /**
     * @brief Get value as real number
     */
    Real get_real(const std::string& key, Real default_val = 0.0) const;

    /**
     * @brief Get value as integer
     */
    Int get_int(const std::string& key, Int default_val = 0) const;

    /**
     * @brief Get value as boolean
     */
    bool get_bool(const std::string& key, bool default_val = false) const;

    /**
     * @brief Get value as real array
     */
    std::vector<Real> get_real_array(const std::string& key) const;

    /**
     * @brief Get value as integer array
     */
    std::vector<Int> get_int_array(const std::string& key) const;

    /**
     * @brief Get value as string array
     */
    std::vector<std::string> get_string_array(const std::string& key) const;

    /**
     * @brief Set value
     */
    void set(const std::string& key, const ConfigValue& value) {
        values_[key] = value;
    }

    /**
     * @brief Get all keys
     */
    std::vector<std::string> keys() const;

    /**
     * @brief Get subsection
     */
    ConfigSection& subsection(const std::string& name);

    /**
     * @brief Get subsection (const)
     */
    const ConfigSection& subsection(const std::string& name) const;

    /**
     * @brief Check if subsection exists
     */
    bool has_subsection(const std::string& name) const {
        return subsections_.count(name) > 0;
    }

    /**
     * @brief Get all subsection names
     */
    std::vector<std::string> subsection_names() const;

private:
    std::string name_;
    std::map<std::string, ConfigValue> values_;
    std::map<std::string, std::shared_ptr<ConfigSection>> subsections_;
};

/**
 * @brief Simple configuration file reader
 */
class ConfigReader {
public:
    ConfigReader() = default;

    /**
     * @brief Read configuration from file
     */
    ConfigSection read(const std::string& filename);

    /**
     * @brief Read configuration from string
     */
    ConfigSection read_string(const std::string& content);

private:
    /**
     * @brief Parse configuration content
     */
    void parse(const std::string& content, ConfigSection& root);

    /**
     * @brief Parse a line
     */
    void parse_line(const std::string& line, ConfigSection& current, int indent_level);

    /**
     * @brief Parse value from string
     */
    ConfigValue parse_value(const std::string& value_str);

    /**
     * @brief Get indent level
     */
    int get_indent_level(const std::string& line) const;

    /**
     * @brief Trim whitespace
     */
    std::string trim(const std::string& str) const;

    std::vector<ConfigSection*> section_stack_;
    std::vector<int> indent_stack_;
};

} // namespace io
} // namespace nxs
