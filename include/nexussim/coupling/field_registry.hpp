#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/data/field.hpp>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace nxs {
namespace coupling {

// Type alias for convenience
using FieldPtr = std::shared_ptr<Field<Real>>;

// ============================================================================
// Field Metadata
// ============================================================================

/**
 * @brief Metadata about a field in the registry
 */
struct FieldMetadata {
    std::string name;             ///< Field name
    std::string provider;         ///< Module that provides this field
    std::vector<std::string> consumers;  ///< Modules that consume this field
    FieldType type;               ///< Field type
    FieldLocation location;       ///< Field location (node/cell/gauss)
    std::size_t size;             ///< Field size
    bool is_shared;               ///< Whether field is shared between modules
    bool is_synchronized;         ///< Whether field is synchronized across MPI ranks
};

// ============================================================================
// Field Registry
// ============================================================================

/**
 * @brief Registry for managing fields across multiple physics modules
 *
 * The FieldRegistry manages field data exchange between different physics
 * modules in a multi-physics simulation. It tracks which modules provide
 * and consume each field, and facilitates data transfer.
 */
class FieldRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static FieldRegistry& instance() {
        static FieldRegistry registry;
        return registry;
    }

    // ========================================================================
    // Field Registration
    // ========================================================================

    /**
     * @brief Register a field provided by a module
     * @param name Field name
     * @param provider Module name that provides this field
     * @param field Shared pointer to the field
     */
    void register_field(const std::string& name,
                       const std::string& provider,
                       FieldPtr field) {
        if (fields_.find(name) != fields_.end()) {
            NXS_LOG_WARN("Field '{}' already registered, overwriting", name);
        }

        fields_[name] = field;

        FieldMetadata meta;
        meta.name = name;
        meta.provider = provider;
        meta.type = field->type();
        meta.location = field->location();
        meta.size = field->size();
        meta.is_shared = false;
        meta.is_synchronized = false;

        metadata_[name] = meta;

        NXS_LOG_INFO("Registered field '{}' from module '{}'", name, provider);
    }

    /**
     * @brief Register a consumer for a field
     * @param name Field name
     * @param consumer Module name that consumes this field
     */
    void register_consumer(const std::string& name,
                          const std::string& consumer) {
        if (metadata_.find(name) == metadata_.end()) {
            throw InvalidArgumentError("Field '" + name + "' not found in registry");
        }

        metadata_[name].consumers.push_back(consumer);
        metadata_[name].is_shared = true;

        NXS_LOG_INFO("Module '{}' registered as consumer of field '{}'",
                    consumer, name);
    }

    // ========================================================================
    // Field Access
    // ========================================================================

    /**
     * @brief Get a field by name
     * @param name Field name
     * @return Shared pointer to field
     */
    FieldPtr get_field(const std::string& name) {
        auto it = fields_.find(name);
        if (it == fields_.end()) {
            throw InvalidArgumentError("Field '" + name + "' not found in registry");
        }
        return it->second;
    }

    /**
     * @brief Check if field exists
     * @param name Field name
     * @return True if field exists
     */
    bool has_field(const std::string& name) const {
        return fields_.find(name) != fields_.end();
    }

    /**
     * @brief Get field metadata
     * @param name Field name
     * @return Field metadata
     */
    const FieldMetadata& get_metadata(const std::string& name) const {
        auto it = metadata_.find(name);
        if (it == metadata_.end()) {
            throw InvalidArgumentError("Field '" + name + "' not found in registry");
        }
        return it->second;
    }

    /**
     * @brief Get all field names
     * @return Vector of field names
     */
    std::vector<std::string> get_field_names() const {
        std::vector<std::string> names;
        names.reserve(fields_.size());
        for (const auto& pair : fields_) {
            names.push_back(pair.first);
        }
        return names;
    }

    /**
     * @brief Get fields provided by a module
     * @param provider Module name
     * @return Vector of field names
     */
    std::vector<std::string> get_provided_fields(const std::string& provider) const {
        std::vector<std::string> names;
        for (const auto& pair : metadata_) {
            if (pair.second.provider == provider) {
                names.push_back(pair.first);
            }
        }
        return names;
    }

    /**
     * @brief Get fields consumed by a module
     * @param consumer Module name
     * @return Vector of field names
     */
    std::vector<std::string> get_consumed_fields(const std::string& consumer) const {
        std::vector<std::string> names;
        for (const auto& pair : metadata_) {
            const auto& consumers = pair.second.consumers;
            if (std::find(consumers.begin(), consumers.end(), consumer) != consumers.end()) {
                names.push_back(pair.first);
            }
        }
        return names;
    }

    // ========================================================================
    // Field Synchronization
    // ========================================================================

    /**
     * @brief Mark field as synchronized
     * @param name Field name
     */
    void mark_synchronized(const std::string& name) {
        auto it = metadata_.find(name);
        if (it != metadata_.end()) {
            it->second.is_synchronized = true;
        }
    }

    /**
     * @brief Mark all fields as unsynchronized
     */
    void clear_sync_flags() {
        for (auto& pair : metadata_) {
            pair.second.is_synchronized = false;
        }
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /**
     * @brief Clear all registered fields
     */
    void clear() {
        fields_.clear();
        metadata_.clear();
        NXS_LOG_INFO("Field registry cleared");
    }

    /**
     * @brief Print registry contents
     */
    void print() const {
        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("Field Registry Contents ({} fields)", fields_.size());
        NXS_LOG_INFO("=================================================");

        for (const auto& pair : metadata_) {
            const auto& meta = pair.second;
            NXS_LOG_INFO("Field: {}", meta.name);
            NXS_LOG_INFO("  Provider: {}", meta.provider);
            NXS_LOG_INFO("  Consumers: {}", meta.consumers.size());
            for (const auto& consumer : meta.consumers) {
                NXS_LOG_INFO("    - {}", consumer);
            }
            NXS_LOG_INFO("  Type: {}", static_cast<int>(meta.type));
            NXS_LOG_INFO("  Location: {}", static_cast<int>(meta.location));
            NXS_LOG_INFO("  Size: {}", meta.size);
            NXS_LOG_INFO("  Shared: {}", meta.is_shared);
        }

        NXS_LOG_INFO("=================================================");
    }

private:
    FieldRegistry() = default;
    ~FieldRegistry() = default;

    // Prevent copying
    FieldRegistry(const FieldRegistry&) = delete;
    FieldRegistry& operator=(const FieldRegistry&) = delete;

    // Storage
    std::unordered_map<std::string, FieldPtr> fields_;
    std::unordered_map<std::string, FieldMetadata> metadata_;
};

} // namespace coupling
} // namespace nxs
