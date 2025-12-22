/**
 * @file config_reader.cpp
 * @brief Configuration reader implementation
 */

#include <nexussim/io/config_reader.hpp>
#include <nexussim/core/logger.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace nxs {
namespace io {

// ============================================================================
// ConfigSection Implementation
// ============================================================================

std::string ConfigSection::get_string(const std::string& key, const std::string& default_val) const {
    if (!has(key)) return default_val;

    const auto& val = values_.at(key);
    if (std::holds_alternative<std::string>(val)) {
        return std::get<std::string>(val);
    }
    return default_val;
}

Real ConfigSection::get_real(const std::string& key, Real default_val) const {
    if (!has(key)) return default_val;

    const auto& val = values_.at(key);
    if (std::holds_alternative<Real>(val)) {
        return std::get<Real>(val);
    } else if (std::holds_alternative<Int>(val)) {
        return static_cast<Real>(std::get<Int>(val));
    }
    return default_val;
}

Int ConfigSection::get_int(const std::string& key, Int default_val) const {
    if (!has(key)) return default_val;

    const auto& val = values_.at(key);
    if (std::holds_alternative<Int>(val)) {
        return std::get<Int>(val);
    } else if (std::holds_alternative<Real>(val)) {
        return static_cast<Int>(std::get<Real>(val));
    }
    return default_val;
}

bool ConfigSection::get_bool(const std::string& key, bool default_val) const {
    if (!has(key)) return default_val;

    const auto& val = values_.at(key);
    if (std::holds_alternative<bool>(val)) {
        return std::get<bool>(val);
    }
    return default_val;
}

std::vector<Real> ConfigSection::get_real_array(const std::string& key) const {
    if (!has(key)) return {};

    const auto& val = values_.at(key);
    if (std::holds_alternative<std::vector<Real>>(val)) {
        return std::get<std::vector<Real>>(val);
    } else if (std::holds_alternative<std::vector<Int>>(val)) {
        const auto& int_vec = std::get<std::vector<Int>>(val);
        std::vector<Real> real_vec(int_vec.size());
        for (std::size_t i = 0; i < int_vec.size(); ++i) {
            real_vec[i] = static_cast<Real>(int_vec[i]);
        }
        return real_vec;
    }
    return {};
}

std::vector<Int> ConfigSection::get_int_array(const std::string& key) const {
    if (!has(key)) return {};

    const auto& val = values_.at(key);
    if (std::holds_alternative<std::vector<Int>>(val)) {
        return std::get<std::vector<Int>>(val);
    }
    return {};
}

std::vector<std::string> ConfigSection::get_string_array(const std::string& key) const {
    if (!has(key)) return {};

    const auto& val = values_.at(key);
    if (std::holds_alternative<std::vector<std::string>>(val)) {
        return std::get<std::vector<std::string>>(val);
    }
    return {};
}

std::vector<std::string> ConfigSection::keys() const {
    std::vector<std::string> result;
    result.reserve(values_.size());
    for (const auto& [key, _] : values_) {
        result.push_back(key);
    }
    return result;
}

ConfigSection& ConfigSection::subsection(const std::string& name) {
    if (!has_subsection(name)) {
        subsections_[name] = std::make_shared<ConfigSection>(name);
    }
    return *subsections_[name];
}

const ConfigSection& ConfigSection::subsection(const std::string& name) const {
    if (!has_subsection(name)) {
        throw InvalidArgumentError("Subsection not found: " + name);
    }
    return *subsections_.at(name);
}

std::vector<std::string> ConfigSection::subsection_names() const {
    std::vector<std::string> result;
    result.reserve(subsections_.size());
    for (const auto& [name, _] : subsections_) {
        result.push_back(name);
    }
    return result;
}

// ============================================================================
// ConfigReader Implementation
// ============================================================================

ConfigSection ConfigReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileIOError("Failed to open config file: " + filename);
    }

    NXS_LOG_INFO("Reading config file: {}", filename);

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    return read_string(buffer.str());
}

ConfigSection ConfigReader::read_string(const std::string& content) {
    ConfigSection root("root");
    parse(content, root);
    return root;
}

void ConfigReader::parse(const std::string& content, ConfigSection& root) {
    std::istringstream stream(content);
    std::string line;

    ConfigSection* current_section = &root;
    int current_indent = 0;
    int list_item_count = 0;  // Track list item index

    section_stack_.clear();
    indent_stack_.clear();
    section_stack_.push_back(&root);
    indent_stack_.push_back(0);

    while (std::getline(stream, line)) {
        // Skip empty lines and comments
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        int indent = get_indent_level(line);

        // Handle indentation changes
        while (!indent_stack_.empty() && indent < indent_stack_.back()) {
            indent_stack_.pop_back();
            section_stack_.pop_back();
            list_item_count = 0;  // Reset list counter when going back
        }

        if (!section_stack_.empty()) {
            current_section = section_stack_.back();
            current_indent = indent_stack_.empty() ? 0 : indent_stack_.back();
        }

        // Check for list items FIRST (before checking for colons)
        if (trimmed[0] == '-' && trimmed.size() > 1 && std::isspace(trimmed[1])) {
            // List item - create indexed subsection (must have space after dash)
            std::string item_content = trim(trimmed.substr(1));

            // Create a subsection for this list item with numeric name
            std::string item_name = std::to_string(list_item_count++);
            NXS_LOG_INFO("Creating list item '{}' under '{}': {}", item_name, current_section->name(), item_content);
            ConfigSection& item_section = current_section->subsection(item_name);

            // If the list item has inline key-value, parse it
            std::size_t item_colon = item_content.find(':');
            if (item_colon != std::string::npos) {
                std::string item_key = trim(item_content.substr(0, item_colon));
                std::string item_value = trim(item_content.substr(item_colon + 1));

                if (!item_value.empty()) {
                    ConfigValue value = parse_value(item_value);
                    item_section.set(item_key, value);
                }
            }

            // Push this item section onto the stack for nested properties
            section_stack_.push_back(&item_section);
            indent_stack_.push_back(indent);
            continue;  // Move to next line
        }

        // Parse key-value pairs
        std::size_t colon_pos = trimmed.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = trim(trimmed.substr(0, colon_pos));
            std::string value_str = trim(trimmed.substr(colon_pos + 1));

            if (value_str.empty()) {
                // This is a section header
                // If we're creating a new section at the SAME indent level as current,
                // and we're not at root level, pop back to parent first
                if (section_stack_.size() > 1 && !indent_stack_.empty() &&
                    indent == indent_stack_.back()) {
                    indent_stack_.pop_back();
                    section_stack_.pop_back();
                    if (!section_stack_.empty()) {
                        current_section = section_stack_.back();
                    }
                }

                NXS_LOG_DEBUG("Creating subsection '{}' under '{}'", key, current_section->name());
                ConfigSection& new_section = current_section->subsection(key);
                section_stack_.push_back(&new_section);
                indent_stack_.push_back(indent);
                list_item_count = 0;  // Reset list counter for new section
            } else {
                // This is a key-value pair
                ConfigValue value = parse_value(value_str);
                current_section->set(key, value);
            }
        }
    }
}

ConfigValue ConfigReader::parse_value(const std::string& value_str) {
    std::string trimmed = trim(value_str);

    // Remove quotes if present
    if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
        return trimmed.substr(1, trimmed.size() - 2);
    }
    if (trimmed.size() >= 2 && trimmed.front() == '\'' && trimmed.back() == '\'') {
        return trimmed.substr(1, trimmed.size() - 2);
    }

    // Check for array [...]
    if (!trimmed.empty() && trimmed.front() == '[' && trimmed.back() == ']') {
        std::string array_str = trimmed.substr(1, trimmed.size() - 2);
        std::vector<Real> real_array;
        std::vector<Int> int_array;
        bool is_int = true;

        std::istringstream iss(array_str);
        std::string item;
        while (std::getline(iss, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;

            try {
                if (item.find('.') != std::string::npos || item.find('e') != std::string::npos) {
                    Real val = std::stod(item);
                    real_array.push_back(val);
                    is_int = false;
                } else {
                    Int val = std::stoi(item);
                    int_array.push_back(val);
                    real_array.push_back(static_cast<Real>(val));
                }
            } catch (...) {
                NXS_LOG_WARN("Failed to parse array element: {}", item);
            }
        }

        if (is_int && !int_array.empty()) {
            return int_array;
        } else if (!real_array.empty()) {
            return real_array;
        }
    }

    // Boolean
    if (trimmed == "true" || trimmed == "True" || trimmed == "TRUE") {
        return true;
    }
    if (trimmed == "false" || trimmed == "False" || trimmed == "FALSE") {
        return false;
    }

    // Try to parse as number
    try {
        if (trimmed.find('.') != std::string::npos ||
            trimmed.find('e') != std::string::npos ||
            trimmed.find('E') != std::string::npos) {
            return std::stod(trimmed);
        } else {
            return static_cast<Int>(std::stoi(trimmed));
        }
    } catch (...) {
        // Not a number, return as string
        return trimmed;
    }
}

int ConfigReader::get_indent_level(const std::string& line) const {
    int count = 0;
    for (char c : line) {
        if (c == ' ') {
            count++;
        } else if (c == '\t') {
            count += 4;  // Treat tab as 4 spaces
        } else {
            break;
        }
    }
    return count;
}

std::string ConfigReader::trim(const std::string& str) const {
    if (str.empty()) return str;

    std::size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";

    std::size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

} // namespace io
} // namespace nxs
