#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/core/exception.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// Element Type Enumerations
// ============================================================================

enum class ElementType {
    // 3D Solid Elements
    Hex8,      ///< 8-node hexahedron
    Hex20,     ///< 20-node hexahedron (quadratic)
    Tet4,      ///< 4-node tetrahedron
    Tet10,     ///< 10-node tetrahedron (quadratic)
    Wedge6,    ///< 6-node wedge/prism

    // 2D Shell Elements
    Shell4,    ///< 4-node shell
    Shell3,    ///< 3-node shell

    // 1D Beam/Truss Elements
    Beam2,     ///< 2-node beam
    Truss2,    ///< 2-node truss

    // Discrete Elements
    Spring,    ///< 2-node spring
    Damper,    ///< 2-node damper
    SpringDamper ///< 2-node spring-damper
};

enum class ElementTopology {
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Wedge,
    Pyramid
};

// ============================================================================
// Element Base Class
// ============================================================================

/**
 * @brief Base class for finite elements
 *
 * This class provides the interface for element-level computations including
 * shape functions, derivatives, integration, and internal force calculation.
 * Designed to be GPU-compatible with Kokkos.
 */
class Element {
public:
    /**
     * @brief Element properties structure
     */
    struct Properties {
        ElementType type;
        ElementTopology topology;
        int num_nodes;          ///< Number of nodes per element
        int num_gauss_points;   ///< Number of Gauss integration points
        int num_dof_per_node;   ///< Degrees of freedom per node
        int spatial_dim;        ///< Spatial dimension (1D, 2D, 3D)
    };

    virtual ~Element() = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    virtual Properties properties() const = 0;

    ElementType type() const { return properties().type; }
    int num_nodes() const { return properties().num_nodes; }
    int num_gauss_points() const { return properties().num_gauss_points; }
    int spatial_dim() const { return properties().spatial_dim; }

    // ========================================================================
    // Shape Functions (GPU-compatible)
    // ========================================================================

    /**
     * @brief Evaluate shape functions at natural coordinates
     * @param xi Natural coordinates (ξ, η, ζ)
     * @param N Output: shape function values [num_nodes]
     */
    KOKKOS_INLINE_FUNCTION
    virtual void shape_functions(const Real xi[3], Real* N) const = 0;

    /**
     * @brief Evaluate shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates (ξ, η, ζ)
     * @param dN Output: derivatives [num_nodes x spatial_dim]
     *           dN[i*spatial_dim + j] = ∂N_i/∂ξ_j
     */
    KOKKOS_INLINE_FUNCTION
    virtual void shape_derivatives(const Real xi[3], Real* dN) const = 0;

    // ========================================================================
    // Geometric Mapping
    // ========================================================================

    /**
     * @brief Compute Jacobian matrix at natural coordinates
     * @param xi Natural coordinates
     * @param coords Nodal coordinates [num_nodes x spatial_dim]
     * @param J Output: Jacobian matrix [spatial_dim x spatial_dim]
     * @return Jacobian determinant
     */
    KOKKOS_INLINE_FUNCTION
    virtual Real jacobian(const Real xi[3],
                          const Real* coords,
                          Real* J) const = 0;

    /**
     * @brief Compute B-matrix (strain-displacement matrix)
     * @param xi Natural coordinates
     * @param coords Nodal coordinates
     * @param B Output: B-matrix [6 x (num_nodes * spatial_dim)]
     *          For 3D: [εxx, εyy, εzz, γxy, γyz, γxz]
     */
    KOKKOS_INLINE_FUNCTION
    virtual void strain_displacement_matrix(const Real xi[3],
                                            const Real* coords,
                                            Real* B) const = 0;

    // ========================================================================
    // Integration
    // ========================================================================

    /**
     * @brief Get Gauss quadrature points and weights
     * @param points Output: Gauss points [num_gauss_points x spatial_dim]
     * @param weights Output: Gauss weights [num_gauss_points]
     */
    virtual void gauss_quadrature(Real* points, Real* weights) const = 0;

    // ========================================================================
    // Element Computations
    // ========================================================================

    /**
     * @brief Compute element mass matrix
     * @param coords Nodal coordinates
     * @param density Material density
     * @param M Output: Element mass matrix [ndof x ndof]
     */
    virtual void mass_matrix(const Real* coords,
                            Real density,
                            Real* M) const = 0;

    /**
     * @brief Compute element stiffness matrix (linear elastic)
     * @param coords Nodal coordinates
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param K Output: Element stiffness matrix [ndof x ndof]
     */
    virtual void stiffness_matrix(const Real* coords,
                                  Real E,
                                  Real nu,
                                  Real* K) const = 0;

    /**
     * @brief Compute internal force vector
     * @param coords Nodal coordinates
     * @param disp Nodal displacements
     * @param stress Stress tensor at integration points
     * @param fint Output: Internal force vector [ndof]
     */
    KOKKOS_INLINE_FUNCTION
    virtual void internal_force(const Real* coords,
                                const Real* disp,
                                const Real* stress,
                                Real* fint) const = 0;

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * @brief Check if a point is inside the element
     * @param coords Nodal coordinates
     * @param point Physical point
     * @param xi Output: Natural coordinates if inside
     * @return True if point is inside element
     */
    virtual bool contains_point(const Real* coords,
                                const Real* point,
                                Real* xi) const = 0;

    /**
     * @brief Compute element volume/area
     * @param coords Nodal coordinates
     * @return Element volume (3D) or area (2D)
     */
    virtual Real volume(const Real* coords) const = 0;

    /**
     * @brief Compute element characteristic length (for time step)
     * @param coords Nodal coordinates
     * @return Characteristic length
     */
    virtual Real characteristic_length(const Real* coords) const = 0;
};

// ============================================================================
// Element Factory
// ============================================================================

/**
 * @brief Factory for creating element objects
 */
class ElementFactory {
public:
    /**
     * @brief Create an element of specified type
     * @param type Element type
     * @return Unique pointer to element
     */
    static std::unique_ptr<Element> create(ElementType type);

    /**
     * @brief Get element type from string
     * @param type_str Element type string (e.g., "Hex8", "Tet4")
     * @return Element type
     */
    static ElementType from_string(const std::string& type_str);

    /**
     * @brief Get string representation of element type
     * @param type Element type
     * @return String representation
     */
    static std::string to_string(ElementType type) {
        switch (type) {
            case ElementType::Hex8: return "Hex8";
            case ElementType::Hex20: return "Hex20";
            case ElementType::Tet4: return "Tet4";
            case ElementType::Tet10: return "Tet10";
            case ElementType::Wedge6: return "Wedge6";
            case ElementType::Shell4: return "Shell4";
            case ElementType::Shell3: return "Shell3";
            case ElementType::Beam2: return "Beam2";
            case ElementType::Truss2: return "Truss2";
            case ElementType::Spring: return "Spring";
            case ElementType::Damper: return "Damper";
            case ElementType::SpringDamper: return "SpringDamper";
            default: return "Unknown";
        }
    }
};

} // namespace physics
} // namespace nxs
