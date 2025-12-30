#pragma once

#include <nexussim/core/core.hpp>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace nxs {

// ============================================================================
// Field - Generic Container for Simulation Data
// ============================================================================

/**
 * @brief Field stores simulation data (scalar, vector, or tensor) at specific locations
 *
 * Fields use Structure-of-Arrays (SOA) layout for optimal vectorization and GPU performance.
 * For example, a vector field stores all X components contiguously, then all Y, then all Z.
 */
template<typename T = Real>
class Field {
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    Field() = default;

    Field(std::string name,
          FieldType type,
          FieldLocation location,
          std::size_t num_entities,
          std::size_t num_components = 1)
        : name_(std::move(name))
        , type_(type)
        , location_(location)
        , num_entities_(num_entities)
        , num_components_(num_components)
        , data_(num_entities * num_components)
    {
        NXS_REQUIRE(num_components > 0, "Number of components must be positive");
        NXS_REQUIRE(num_entities > 0, "Number of entities must be positive");

        // Validate component count matches field type
        validate_components();
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    const std::string& name() const { return name_; }
    FieldType type() const { return type_; }
    FieldLocation location() const { return location_; }
    std::size_t num_entities() const { return num_entities_; }
    std::size_t num_components() const { return num_components_; }
    std::size_t size() const { return data_.size(); }

    // Raw data access
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Get span for specific component (SOA layout)
    std::span<T> component(std::size_t comp) {
        NXS_CHECK_RANGE(comp, num_components_);
        return std::span<T>(data_.data() + comp * num_entities_, num_entities_);
    }

    std::span<const T> component(std::size_t comp) const {
        NXS_CHECK_RANGE(comp, num_components_);
        return std::span<const T>(data_.data() + comp * num_entities_, num_entities_);
    }

    // ========================================================================
    // Scalar Field Access (num_components == 1)
    // ========================================================================

    T& operator[](std::size_t i) {
        NXS_DEBUG_ASSERT(num_components_ == 1, "Use component() for multi-component fields");
        NXS_CHECK_RANGE(i, num_entities_);
        return data_[i];
    }

    const T& operator[](std::size_t i) const {
        NXS_DEBUG_ASSERT(num_components_ == 1, "Use component() for multi-component fields");
        NXS_CHECK_RANGE(i, num_entities_);
        return data_[i];
    }

    // ========================================================================
    // Multi-Component Access (entity_id, component_id)
    // ========================================================================

    T& at(std::size_t entity_id, std::size_t comp_id) {
        NXS_CHECK_RANGE(entity_id, num_entities_);
        NXS_CHECK_RANGE(comp_id, num_components_);
        return data_[comp_id * num_entities_ + entity_id];
    }

    const T& at(std::size_t entity_id, std::size_t comp_id) const {
        NXS_CHECK_RANGE(entity_id, num_entities_);
        NXS_CHECK_RANGE(comp_id, num_components_);
        return data_[comp_id * num_entities_ + entity_id];
    }

    // ========================================================================
    // Vector Operations
    // ========================================================================

    // Get vector at entity (for vector fields)
    template<std::size_t N>
    std::array<T, N> get_vec(std::size_t entity_id) const {
        NXS_REQUIRE(N == num_components_, "Size mismatch");
        NXS_CHECK_RANGE(entity_id, num_entities_);

        std::array<T, N> result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = data_[i * num_entities_ + entity_id];
        }
        return result;
    }

    // Set vector at entity
    template<std::size_t N>
    void set_vec(std::size_t entity_id, const std::array<T, N>& vec) {
        NXS_REQUIRE(N == num_components_, "Size mismatch");
        NXS_CHECK_RANGE(entity_id, num_entities_);

        for (std::size_t i = 0; i < N; ++i) {
            data_[i * num_entities_ + entity_id] = vec[i];
        }
    }

    // ========================================================================
    // Utility Operations
    // ========================================================================

    // Fill with value
    void fill(const T& value) {
        data_.fill(value);
    }

    // Zero all data
    void zero() {
        data_.zero();
    }

    // Resize (warning: does not preserve data)
    void resize(std::size_t new_num_entities) {
        num_entities_ = new_num_entities;
        data_.resize(num_entities_ * num_components_);
    }

    // Copy data from another field
    void copy_from(const Field<T>& other) {
        NXS_REQUIRE(type_ == other.type_, "Field types must match");
        NXS_REQUIRE(location_ == other.location_, "Field locations must match");
        NXS_REQUIRE(num_components_ == other.num_components_, "Component counts must match");

        if (num_entities_ != other.num_entities_) {
            resize(other.num_entities_);
        }

        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

    // Compute statistics
    T min() const {
        if (data_.size() == 0) return T{};
        return *std::min_element(data_.begin(), data_.end());
    }

    T max() const {
        if (data_.size() == 0) return T{};
        return *std::max_element(data_.begin(), data_.end());
    }

    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }

    T mean() const {
        if (data_.size() == 0) return T{};
        return sum() / static_cast<T>(data_.size());
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    // Set/get metadata string
    void set_description(const std::string& desc) { description_ = desc; }
    const std::string& description() const { return description_; }

    void set_units(const std::string& units) { units_ = units; }
    const std::string& units() const { return units_; }

private:
    void validate_components() {
        switch (type_) {
            case FieldType::Scalar:
                NXS_REQUIRE(num_components_ == 1, "Scalar fields must have 1 component");
                break;
            case FieldType::Vector:
                NXS_REQUIRE(num_components_ >= 2 && num_components_ <= 4,
                           "Vector fields must have 2-4 components");
                break;
            case FieldType::Tensor:
                NXS_REQUIRE(num_components_ == 6 || num_components_ == 9,
                           "Tensor fields must have 6 (Voigt) or 9 (full) components");
                break;
        }
    }

    std::string name_;
    std::string description_;
    std::string units_;
    FieldType type_;
    FieldLocation location_;
    std::size_t num_entities_;
    std::size_t num_components_;
    AlignedBuffer<T> data_;
};

// ============================================================================
// Type Aliases for Common Field Types
// ============================================================================

using ScalarField = Field<Real>;
using VectorField = Field<Real>;
using TensorField = Field<Real>;

using ScalarFieldi = Field<Int>;
using VectorFieldi = Field<Int>;

// ============================================================================
// Field Factory Functions
// ============================================================================

template<typename T = Real>
inline Field<T> make_scalar_field(const std::string& name,
                                   FieldLocation location,
                                   std::size_t num_entities) {
    return Field<T>(name, FieldType::Scalar, location, num_entities, 1);
}

template<typename T = Real>
inline Field<T> make_vector_field(const std::string& name,
                                   FieldLocation location,
                                   std::size_t num_entities,
                                   std::size_t dim = 3) {
    return Field<T>(name, FieldType::Vector, location, num_entities, dim);
}

template<typename T = Real>
inline Field<T> make_tensor_field(const std::string& name,
                                   FieldLocation location,
                                   std::size_t num_entities,
                                   bool use_voigt = true) {
    std::size_t components = use_voigt ? 6 : 9;
    return Field<T>(name, FieldType::Tensor, location, num_entities, components);
}

} // namespace nxs
