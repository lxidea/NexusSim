#pragma once

/**
 * @file sph_kernel.hpp
 * @brief SPH kernel (smoothing) functions
 *
 * Implements various SPH kernel functions:
 * - Cubic Spline (standard, most common)
 * - Wendland C2/C4 (compact support, no tensile instability)
 * - Gaussian (infinite support, smooth)
 * - Quintic Spline (higher accuracy)
 *
 * Each kernel W(r,h) satisfies:
 * - Normalization: ∫W(r,h)dV = 1
 * - Compact support: W(r,h) = 0 for r > kh
 * - Delta property: W(r,h) → δ(r) as h → 0
 */

#include <nexussim/core/core.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>

namespace nxs {
namespace sph {

// ============================================================================
// Kernel Type Enumeration
// ============================================================================

enum class KernelType {
    CubicSpline,    ///< Standard cubic B-spline (support = 2h)
    WendlandC2,     ///< Wendland C2 quintic (support = 2h)
    WendlandC4,     ///< Wendland C4 (support = 2h)
    Gaussian,       ///< Gaussian (truncated at 3h)
    QuinticSpline   ///< 5th-order B-spline (support = 3h)
};

// ============================================================================
// Kernel Base Functions (KOKKOS_INLINE_FUNCTION for GPU)
// ============================================================================

/**
 * @brief Cubic Spline Kernel (Monaghan 1992)
 *
 * W(q) = σ * { (2-q)³ - 4(1-q)³  for 0 ≤ q < 1
 *            { (2-q)³            for 1 ≤ q < 2
 *            { 0                 for q ≥ 2
 *
 * where q = r/h, σ = 1/(πh³) in 3D
 */
class CubicSplineKernel {
public:
    static constexpr Real support_radius = 2.0;  // W = 0 for r > 2h

    KOKKOS_INLINE_FUNCTION
    static Real W(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t1 = 2.0 - q;
        Real w = t1 * t1 * t1;

        if (q < 1.0) {
            Real t2 = 1.0 - q;
            w -= 4.0 * t2 * t2 * t2;
        }

        return w;
    }

    KOKKOS_INLINE_FUNCTION
    static Real dWdq(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t1 = 2.0 - q;
        Real dw = -3.0 * t1 * t1;

        if (q < 1.0) {
            Real t2 = 1.0 - q;
            dw += 12.0 * t2 * t2;
        }

        return dw;
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_3d(Real h) {
        return 1.0 / (M_PI * h * h * h);
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_2d(Real h) {
        return 10.0 / (7.0 * M_PI * h * h);
    }
};

/**
 * @brief Wendland C2 Kernel (Wendland 1995)
 *
 * W(q) = σ * (1-q/2)⁴ * (2q+1) for q < 2, else 0
 *
 * Properties:
 * - Compact support (2h)
 * - C2 continuous
 * - No tensile instability
 */
class WendlandC2Kernel {
public:
    static constexpr Real support_radius = 2.0;

    KOKKOS_INLINE_FUNCTION
    static Real W(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t = 1.0 - 0.5 * q;
        Real t4 = t * t * t * t;
        return t4 * (2.0 * q + 1.0);
    }

    KOKKOS_INLINE_FUNCTION
    static Real dWdq(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t = 1.0 - 0.5 * q;
        Real t3 = t * t * t;
        // d/dq[(1-q/2)⁴(2q+1)] = -5q(1-q/2)³
        return -5.0 * q * t3;
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_3d(Real h) {
        return 21.0 / (16.0 * M_PI * h * h * h);
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_2d(Real h) {
        return 7.0 / (4.0 * M_PI * h * h);
    }
};

/**
 * @brief Wendland C4 Kernel
 *
 * W(q) = σ * (1-q/2)⁶ * (35q²/12 + 3q + 1) for q < 2, else 0
 *
 * Higher smoothness than C2, better for second derivatives
 */
class WendlandC4Kernel {
public:
    static constexpr Real support_radius = 2.0;

    KOKKOS_INLINE_FUNCTION
    static Real W(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t = 1.0 - 0.5 * q;
        Real t6 = t * t * t * t * t * t;
        return t6 * (35.0 * q * q / 12.0 + 3.0 * q + 1.0);
    }

    KOKKOS_INLINE_FUNCTION
    static Real dWdq(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 2.0) return 0.0;

        Real t = 1.0 - 0.5 * q;
        Real t5 = t * t * t * t * t;
        // Derivative (simplified)
        return -t5 * q * (7.0 * q / 2.0 + 4.0);
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_3d(Real h) {
        return 495.0 / (256.0 * M_PI * h * h * h);
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_2d(Real h) {
        return 9.0 / (4.0 * M_PI * h * h);
    }
};

/**
 * @brief Quintic Spline Kernel
 *
 * W(q) = σ * { (3-q)⁵ - 6(2-q)⁵ + 15(1-q)⁵  for 0 ≤ q < 1
 *            { (3-q)⁵ - 6(2-q)⁵             for 1 ≤ q < 2
 *            { (3-q)⁵                        for 2 ≤ q < 3
 *            { 0                             for q ≥ 3
 *
 * Higher accuracy, larger support radius (3h)
 */
class QuinticSplineKernel {
public:
    static constexpr Real support_radius = 3.0;

    KOKKOS_INLINE_FUNCTION
    static Real W(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 3.0) return 0.0;

        auto pow5 = [](Real x) { return x * x * x * x * x; };

        Real w = pow5(3.0 - q);
        if (q < 2.0) w -= 6.0 * pow5(2.0 - q);
        if (q < 1.0) w += 15.0 * pow5(1.0 - q);

        return w;
    }

    KOKKOS_INLINE_FUNCTION
    static Real dWdq(Real q) {
        if (q < 0.0) q = -q;
        if (q >= 3.0) return 0.0;

        auto pow4 = [](Real x) { return x * x * x * x; };

        Real dw = -5.0 * pow4(3.0 - q);
        if (q < 2.0) dw += 30.0 * pow4(2.0 - q);
        if (q < 1.0) dw -= 75.0 * pow4(1.0 - q);

        return dw;
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_3d(Real h) {
        return 1.0 / (120.0 * M_PI * h * h * h);
    }

    KOKKOS_INLINE_FUNCTION
    static Real normalization_2d(Real h) {
        return 7.0 / (478.0 * M_PI * h * h);
    }
};

// ============================================================================
// Generic Kernel Wrapper
// ============================================================================

/**
 * @brief Generic SPH kernel that dispatches to specific implementations
 */
class SPHKernel {
public:
    SPHKernel(KernelType type = KernelType::CubicSpline, int dim = 3)
        : type_(type), dim_(dim) {}

    /**
     * @brief Get support radius in units of h
     */
    Real support_radius() const {
        switch (type_) {
            case KernelType::CubicSpline:  return CubicSplineKernel::support_radius;
            case KernelType::WendlandC2:   return WendlandC2Kernel::support_radius;
            case KernelType::WendlandC4:   return WendlandC4Kernel::support_radius;
            case KernelType::QuinticSpline: return QuinticSplineKernel::support_radius;
            case KernelType::Gaussian:     return 3.0;  // Truncated
            default: return 2.0;
        }
    }

    /**
     * @brief Evaluate kernel W(r, h)
     */
    KOKKOS_INLINE_FUNCTION
    Real W(Real r, Real h) const {
        Real q = r / h;
        Real sigma = (dim_ == 3) ? normalization_3d(h) : normalization_2d(h);
        return sigma * W_raw(q);
    }

    /**
     * @brief Evaluate kernel gradient magnitude |∇W|
     * Returns dW/dr (multiply by r̂ for vector gradient)
     */
    KOKKOS_INLINE_FUNCTION
    Real grad_W(Real r, Real h) const {
        if (r < 1e-12) return 0.0;  // Avoid singularity
        Real q = r / h;
        Real sigma = (dim_ == 3) ? normalization_3d(h) : normalization_2d(h);
        return sigma * dWdq_raw(q) / h;
    }

    /**
     * @brief Compute kernel gradient vector
     * ∇W = (dW/dr) * (r_ij / |r_ij|)
     */
    KOKKOS_INLINE_FUNCTION
    void grad_W_vec(Real rx, Real ry, Real rz, Real h,
                    Real& gx, Real& gy, Real& gz) const {
        Real r = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (r < 1e-12) {
            gx = gy = gz = 0.0;
            return;
        }

        Real dWdr = grad_W(r, h);
        Real inv_r = 1.0 / r;
        gx = dWdr * rx * inv_r;
        gy = dWdr * ry * inv_r;
        gz = dWdr * rz * inv_r;
    }

    KernelType type() const { return type_; }
    int dimension() const { return dim_; }

private:
    KOKKOS_INLINE_FUNCTION
    Real W_raw(Real q) const {
        switch (type_) {
            case KernelType::CubicSpline:  return CubicSplineKernel::W(q);
            case KernelType::WendlandC2:   return WendlandC2Kernel::W(q);
            case KernelType::WendlandC4:   return WendlandC4Kernel::W(q);
            case KernelType::QuinticSpline: return QuinticSplineKernel::W(q);
            default: return CubicSplineKernel::W(q);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real dWdq_raw(Real q) const {
        switch (type_) {
            case KernelType::CubicSpline:  return CubicSplineKernel::dWdq(q);
            case KernelType::WendlandC2:   return WendlandC2Kernel::dWdq(q);
            case KernelType::WendlandC4:   return WendlandC4Kernel::dWdq(q);
            case KernelType::QuinticSpline: return QuinticSplineKernel::dWdq(q);
            default: return CubicSplineKernel::dWdq(q);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real normalization_3d(Real h) const {
        switch (type_) {
            case KernelType::CubicSpline:  return CubicSplineKernel::normalization_3d(h);
            case KernelType::WendlandC2:   return WendlandC2Kernel::normalization_3d(h);
            case KernelType::WendlandC4:   return WendlandC4Kernel::normalization_3d(h);
            case KernelType::QuinticSpline: return QuinticSplineKernel::normalization_3d(h);
            default: return CubicSplineKernel::normalization_3d(h);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real normalization_2d(Real h) const {
        switch (type_) {
            case KernelType::CubicSpline:  return CubicSplineKernel::normalization_2d(h);
            case KernelType::WendlandC2:   return WendlandC2Kernel::normalization_2d(h);
            case KernelType::WendlandC4:   return WendlandC4Kernel::normalization_2d(h);
            case KernelType::QuinticSpline: return QuinticSplineKernel::normalization_2d(h);
            default: return CubicSplineKernel::normalization_2d(h);
        }
    }

    KernelType type_;
    int dim_;
};

} // namespace sph
} // namespace nxs
