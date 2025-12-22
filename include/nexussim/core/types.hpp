#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <span>

namespace nxs {

// ============================================================================
// Precision Types
// ============================================================================

#ifdef NEXUSSIM_REAL
using Real = NEXUSSIM_REAL;
#else
using Real = double;  // Default to double precision
#endif

// Integer types
using Index = std::size_t;
using Int = std::int32_t;
using Int64 = std::int64_t;

// ============================================================================
// Vector and Matrix Types
// ============================================================================

template<typename T, std::size_t N>
using Array = std::array<T, N>;

// Fixed-size vectors
template<typename T>
using Vec2 = Array<T, 2>;

template<typename T>
using Vec3 = Array<T, 3>;

template<typename T>
using Vec4 = Array<T, 4>;

template<typename T>
using Vec6 = Array<T, 6>;  // For stress/strain tensors

// Common type aliases
using Vec2r = Vec2<Real>;
using Vec3r = Vec3<Real>;
using Vec4r = Vec4<Real>;
using Vec6r = Vec6<Real>;

using Vec2i = Vec2<Int>;
using Vec3i = Vec3<Int>;
using Vec4i = Vec4<Int>;

// ============================================================================
// Smart Pointer Aliases
// ============================================================================

template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

// Factory functions
template<typename T, typename... Args>
inline UniquePtr<T> make_unique(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename... Args>
inline SharedPtr<T> make_shared(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// ============================================================================
// Enumeration Types
// ============================================================================

enum class FieldType {
    Scalar,
    Vector,
    Tensor
};

enum class FieldLocation {
    Node,
    Element,
    IntegrationPoint,
    Particle
};

enum class ElementType {
    // Beams
    Beam2,
    Beam3,

    // Shells
    Shell3,
    Shell4,
    Shell6,

    // Solids
    Tet4,
    Tet10,
    Hex8,
    Hex20,
    Hex27,
    Wedge6,
    Wedge15,

    // Special
    Truss,
    Spring,
    Damper,
    SpringDamper,
    Mass,

    // Meshfree
    SPHParticle,
    RKPMParticle,
    PDNode
};

enum class MaterialType {
    Elastic,
    ElastoPlastic,
    Hyperelastic,
    Viscoplastic,
    Composite,
    Fluid,
    User
};

enum class SolverType {
    Explicit,
    Implicit,
    QuasiStatic
};

enum class TimeIntegrator {
    CentralDifference,
    NewmarkBeta,
    GeneralizedAlpha,
    RungeKutta4
};

enum class LinearSolverType {
    Direct,
    ConjugateGradient,
    GMRES,
    BiCGSTAB
};

enum class ExecutionSpace {
    CPU,
    GPU,
    Auto  // Let runtime decide
};

// ============================================================================
// Constants
// ============================================================================

namespace constants {

template<typename T = Real>
inline constexpr T pi = T(3.14159265358979323846);

template<typename T = Real>
inline constexpr T two_pi = T(2) * pi<T>;

template<typename T = Real>
inline constexpr T half_pi = pi<T> / T(2);

template<typename T = Real>
inline constexpr T sqrt_two = T(1.41421356237309504880);

template<typename T = Real>
inline constexpr T sqrt_three = T(1.73205080756887729352);

// Small number for comparisons
template<typename T = Real>
inline constexpr T epsilon = std::is_same_v<T, float> ? T(1e-6) : T(1e-12);

} // namespace constants

// ============================================================================
// Utility Functions
// ============================================================================

// Convert enum to string (for debugging/logging)
inline const char* to_string(FieldType type) {
    switch (type) {
        case FieldType::Scalar: return "Scalar";
        case FieldType::Vector: return "Vector";
        case FieldType::Tensor: return "Tensor";
        default: return "Unknown";
    }
}

inline const char* to_string(FieldLocation loc) {
    switch (loc) {
        case FieldLocation::Node: return "Node";
        case FieldLocation::Element: return "Element";
        case FieldLocation::IntegrationPoint: return "IntegrationPoint";
        case FieldLocation::Particle: return "Particle";
        default: return "Unknown";
    }
}

inline const char* to_string(ElementType type) {
    switch (type) {
        case ElementType::Beam2: return "Beam2";
        case ElementType::Beam3: return "Beam3";
        case ElementType::Shell3: return "Shell3";
        case ElementType::Shell4: return "Shell4";
        case ElementType::Shell6: return "Shell6";
        case ElementType::Tet4: return "Tet4";
        case ElementType::Tet10: return "Tet10";
        case ElementType::Hex8: return "Hex8";
        case ElementType::Hex20: return "Hex20";
        case ElementType::Hex27: return "Hex27";
        case ElementType::Wedge6: return "Wedge6";
        case ElementType::Wedge15: return "Wedge15";
        case ElementType::Truss: return "Truss";
        case ElementType::Spring: return "Spring";
        case ElementType::Damper: return "Damper";
        case ElementType::SpringDamper: return "SpringDamper";
        case ElementType::Mass: return "Mass";
        case ElementType::SPHParticle: return "SPHParticle";
        case ElementType::RKPMParticle: return "RKPMParticle";
        case ElementType::PDNode: return "PDNode";
        default: return "Unknown";
    }
}

inline const char* to_string(SolverType type) {
    switch (type) {
        case SolverType::Explicit: return "Explicit";
        case SolverType::Implicit: return "Implicit";
        case SolverType::QuasiStatic: return "QuasiStatic";
        default: return "Unknown";
    }
}

} // namespace nxs
