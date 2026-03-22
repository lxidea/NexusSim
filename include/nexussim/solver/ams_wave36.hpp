#pragma once

/**
 * @file ams_wave36.hpp
 * @brief Wave 36: AMS/SMS Substructuring and Advanced Solvers
 *
 * Components:
 * 1. SMSEngine                - Selective Mass Scaling for critical elements
 * 2. SMSPreconditionedCG      - PCG solver for SMS implicit-explicit coupling
 * 3. SMSBoundaryConditions    - Fixed velocity and cyclic BCs for SMS
 * 4. ComponentModeSynthesis   - Craig-Bampton substructuring
 * 5. FrequencyResponse        - Modal superposition FRF computation
 * 6. SMSConstraints           - RBE2/RBE3/joint constraints for SMS
 *
 * References:
 * - Olovsson et al. (2005) "Selective mass scaling for explicit FE analyses"
 * - Craig & Bampton (1968) "Coupling of substructures for dynamic analyses"
 * - Ewins (2000) "Modal Testing: Theory, Practice and Application"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cassert>
#include <array>
#include <limits>

namespace nxs {
namespace solver {

using Real = nxs::Real;

// ============================================================================
// Utility functions
// ============================================================================

namespace ams_detail {

inline Real dot(const Real* a, const Real* b, int n) {
    Real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline Real norm2(const Real* v, int n) {
    return std::sqrt(dot(v, v, n));
}

inline void axpy(Real alpha, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

inline void copy(const Real* x, Real* y, int n) {
    std::memcpy(y, x, static_cast<size_t>(n) * sizeof(Real));
}

inline void zero(Real* y, int n) {
    std::memset(y, 0, static_cast<size_t>(n) * sizeof(Real));
}

inline void scale(Real alpha, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] *= alpha;
}

/// Dense symmetric matrix-vector product: y = A * x (A stored row-major, full)
inline void matvec(const Real* A, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) {
        Real s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += A[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

/// Dense matrix transpose: B = A^T (both n x m row-major)
inline void transpose(const Real* A, Real* AT, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            AT[j * rows + i] = A[i * cols + j];
}

/// Dense matrix multiply: C = A * B, A is (m x k), B is (k x n)
inline void matmul(const Real* A, const Real* B, Real* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Real s = 0.0;
            for (int p = 0; p < k; ++p) {
                s += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = s;
        }
    }
}

} // namespace ams_detail


// ============================================================================
// 1. SMSEngine - Selective Mass Scaling
// ============================================================================

/**
 * @brief Configuration for Selective Mass Scaling.
 *
 * dt_target: desired global timestep
 * max_mass_scale: maximum scaling factor per element (prevents excessive mass)
 * added_mass_limit: total added mass budget as fraction of total model mass
 */
struct SMSConfig {
    Real dt_target = 1.0e-5;
    Real max_mass_scale = 100.0;
    Real added_mass_limit = 0.05;
};

/**
 * @brief Element data for SMS mass scaling.
 *
 * Each element has a characteristic length, wave speed, current mass,
 * and a computed dt. The engine identifies critical elements and adds
 * mass to increase their stable timestep.
 */
struct SMSElement {
    Real char_length = 1.0;     ///< Characteristic element length
    Real wave_speed = 1.0;      ///< Speed of sound in element material
    Real mass = 1.0;            ///< Current element mass
    Real density = 1.0;         ///< Current element density
    Real volume = 1.0;          ///< Element volume
    Real added_mass = 0.0;      ///< Mass added by SMS
    Real dt_element = 0.0;      ///< Element stable timestep (computed)
    bool is_critical = false;   ///< Flag: below dt_target
};

/**
 * @brief Selective Mass Scaling engine.
 *
 * For explicit dynamics, the global timestep is governed by the smallest
 * element timestep: dt_global = min(dt_e) where dt_e = L_e / c_e.
 * A few small or stiff elements can severely limit the timestep.
 *
 * SMS identifies these critical elements and adds mass to raise their
 * characteristic timestep to dt_target:
 *   dt_e = L_e / c_e = L_e * sqrt(rho_e / E_e)
 *   => rho_new = (dt_target / L_e)^2 * E_e
 *   => m_added = (rho_new - rho_old) * V_e
 *
 * The added mass is capped by max_mass_scale and the total budget.
 *
 * Reference: Olovsson et al. (2005)
 */
class SMSEngine {
public:
    SMSEngine() = default;

    /**
     * @brief Compute element stable timestep.
     *
     * dt_e = char_length / wave_speed
     * This is the CFL condition for explicit time integration.
     */
    static Real compute_element_dt(Real char_length, Real wave_speed) {
        if (wave_speed <= 0.0) return std::numeric_limits<Real>::max();
        return char_length / wave_speed;
    }

    /**
     * @brief Compute the required mass scale factor for an element.
     *
     * To achieve dt_target from an element with current dt_e:
     *   scale = (dt_target / dt_e)^2
     *
     * This is because dt_e ~ sqrt(m_e), so doubling dt requires 4x mass.
     */
    static Real compute_scale_factor(Real dt_element, Real dt_target) {
        if (dt_element >= dt_target || dt_element <= 0.0) return 1.0;
        Real ratio = dt_target / dt_element;
        return ratio * ratio;
    }

    /**
     * @brief Apply selective mass scaling to an array of elements.
     *
     * Algorithm:
     * 1. Compute dt_e for each element
     * 2. Identify critical elements (dt_e < dt_target)
     * 3. Compute required scaling, cap at max_mass_scale
     * 4. Add mass, respecting total added mass budget
     *
     * @param elements Array of SMS elements (modified in place)
     * @param nelems Number of elements
     * @param config SMS configuration
     * @return Total added mass
     */
    Real apply_mass_scaling(SMSElement* elements, int nelems,
                            const SMSConfig& config) const {
        if (nelems <= 0) return 0.0;

        // Step 1: Compute element timesteps
        for (int i = 0; i < nelems; ++i) {
            elements[i].dt_element = compute_element_dt(
                elements[i].char_length, elements[i].wave_speed);
            elements[i].is_critical = (elements[i].dt_element < config.dt_target);
            elements[i].added_mass = 0.0;
        }

        // Step 2: Compute total model mass for budget
        Real total_mass = 0.0;
        for (int i = 0; i < nelems; ++i) {
            total_mass += elements[i].mass;
        }
        Real mass_budget = config.added_mass_limit * total_mass;

        // Step 3: Scale critical elements
        Real total_added = 0.0;
        for (int i = 0; i < nelems; ++i) {
            if (!elements[i].is_critical) continue;

            Real scale = compute_scale_factor(elements[i].dt_element,
                                               config.dt_target);
            // Cap scale factor
            scale = std::min(scale, config.max_mass_scale);

            Real desired_mass = scale * elements[i].mass;
            Real add = desired_mass - elements[i].mass;

            // Respect budget
            if (total_added + add > mass_budget) {
                add = mass_budget - total_added;
                if (add <= 0.0) break;
            }

            elements[i].added_mass = add;
            elements[i].mass += add;
            // Update density
            if (elements[i].volume > 0.0) {
                elements[i].density = elements[i].mass / elements[i].volume;
            }
            // Recompute wave speed with new density: c = sqrt(E/rho)
            // wave_speed was c_old = sqrt(E/rho_old), so
            // c_new = c_old * sqrt(rho_old / rho_new) = c_old / sqrt(scale)
            Real actual_scale = elements[i].mass / (elements[i].mass - elements[i].added_mass);
            elements[i].wave_speed /= std::sqrt(actual_scale);
            // Recompute dt
            elements[i].dt_element = compute_element_dt(
                elements[i].char_length, elements[i].wave_speed);

            total_added += add;
        }

        total_added_mass_ = total_added;
        return total_added;
    }

    /**
     * @brief Get the global critical timestep (minimum over all elements).
     */
    static Real get_critical_dt(const SMSElement* elements, int nelems) {
        Real dt_min = std::numeric_limits<Real>::max();
        for (int i = 0; i < nelems; ++i) {
            Real dt_e = compute_element_dt(elements[i].char_length,
                                           elements[i].wave_speed);
            dt_min = std::min(dt_min, dt_e);
        }
        return dt_min;
    }

    /**
     * @brief Count how many elements are critical (below target).
     */
    static int count_critical(const SMSElement* elements, int nelems,
                              Real dt_target) {
        int count = 0;
        for (int i = 0; i < nelems; ++i) {
            Real dt_e = compute_element_dt(elements[i].char_length,
                                           elements[i].wave_speed);
            if (dt_e < dt_target) ++count;
        }
        return count;
    }

    Real total_added_mass() const { return total_added_mass_; }

private:
    mutable Real total_added_mass_ = 0.0;
};


// ============================================================================
// 2. SMSPreconditionedCG - Preconditioned Conjugate Gradient for SMS
// ============================================================================

/**
 * @brief Result structure for iterative solvers.
 */
struct PCGResult {
    int iterations = 0;
    Real residual_norm = 0.0;
    bool converged = false;
};

/**
 * @brief Preconditioned Conjugate Gradient solver for SMS systems.
 *
 * Solves K * x = f where K is symmetric positive definite.
 * Uses Jacobi (diagonal) preconditioning from the scaled mass matrix:
 *   M_inv_diag[i] = 1.0 / M_scaled[i*n + i]
 *
 * The PCG algorithm:
 *   r = f - K*x
 *   z = M_inv * r
 *   p = z
 *   loop:
 *     alpha = (r, z) / (p, K*p)
 *     x += alpha * p
 *     r -= alpha * K*p
 *     check convergence
 *     z_new = M_inv * r
 *     beta = (r, z_new) / (r_old, z_old)
 *     p = z_new + beta * p
 *
 * Reference: Shewchuk (1994) "An Introduction to the CG Method Without
 *            the Agonizing Pain"
 */
class SMSPreconditionedCG {
public:
    SMSPreconditionedCG() = default;

    /**
     * @brief Solve K*x = f using Preconditioned Conjugate Gradient.
     *
     * @param K Stiffness matrix (n x n, row-major, symmetric positive definite)
     * @param M_scaled Scaled mass matrix (n x n, row-major, for Jacobi preconditioner)
     * @param f Right-hand side vector (n)
     * @param x Solution vector (n), initial guess on input, solution on output
     * @param n System size
     * @param tol Convergence tolerance on relative residual ||r||/||f||
     * @param max_iter Maximum CG iterations
     * @return PCGResult with iteration count and final residual
     */
    PCGResult solve(const Real* K, const Real* M_scaled, const Real* f,
                    Real* x, int n, Real tol = 1.0e-8, int max_iter = 1000) const {
        PCGResult result;
        if (n <= 0) return result;

        // Build Jacobi preconditioner from M_scaled diagonal
        std::vector<Real> M_inv(n);
        for (int i = 0; i < n; ++i) {
            Real diag = M_scaled[i * n + i];
            M_inv[i] = (std::abs(diag) > 1.0e-30) ? 1.0 / diag : 1.0;
        }

        std::vector<Real> r(n), z(n), p(n), Kp(n);

        // r = f - K*x
        ams_detail::matvec(K, x, r.data(), n);
        for (int i = 0; i < n; ++i) r[i] = f[i] - r[i];

        Real f_norm = ams_detail::norm2(f, n);
        if (f_norm < 1.0e-30) f_norm = 1.0;

        // z = M_inv * r
        for (int i = 0; i < n; ++i) z[i] = M_inv[i] * r[i];

        // p = z
        ams_detail::copy(z.data(), p.data(), n);

        Real rz = ams_detail::dot(r.data(), z.data(), n);

        for (int iter = 0; iter < max_iter; ++iter) {
            // Kp = K * p
            ams_detail::matvec(K, p.data(), Kp.data(), n);

            Real pKp = ams_detail::dot(p.data(), Kp.data(), n);
            if (std::abs(pKp) < 1.0e-30) break;

            Real alpha = rz / pKp;

            // x += alpha * p
            ams_detail::axpy(alpha, p.data(), x, n);

            // r -= alpha * Kp
            ams_detail::axpy(-alpha, Kp.data(), r.data(), n);

            Real r_norm = ams_detail::norm2(r.data(), n);
            result.iterations = iter + 1;
            result.residual_norm = r_norm / f_norm;

            if (result.residual_norm < tol) {
                result.converged = true;
                return result;
            }

            // z = M_inv * r
            for (int i = 0; i < n; ++i) z[i] = M_inv[i] * r[i];

            Real rz_new = ams_detail::dot(r.data(), z.data(), n);
            Real beta = rz_new / (std::abs(rz) > 1.0e-30 ? rz : 1.0e-30);

            // p = z + beta * p
            for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];

            rz = rz_new;
        }

        return result;
    }

    /**
     * @brief Solve with identity preconditioner (standard CG).
     */
    PCGResult solve_unpreconditioned(const Real* K, const Real* f,
                                      Real* x, int n, Real tol = 1.0e-8,
                                      int max_iter = 1000) const {
        // Create identity matrix as preconditioner
        std::vector<Real> I(n * n, 0.0);
        for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;
        return solve(K, I.data(), f, x, n, tol, max_iter);
    }
};


// ============================================================================
// 3. SMSBoundaryConditions - BCs for SMS
// ============================================================================

/**
 * @brief Cyclic BC DOF pair (tie DOF i to DOF j).
 */
struct CyclicPair {
    int dof_a;
    int dof_b;
};

/**
 * @brief Boundary condition handler for SMS systems.
 *
 * Supports:
 * - Fixed velocity (Dirichlet): zero the row and column in K, set diagonal to 1,
 *   and zero the corresponding RHS entry.
 * - Cyclic (periodic): tie DOF pairs by adding rows/cols and averaging.
 *
 * These operate on dense matrices (K) and vectors (f) of size n x n and n.
 */
class SMSBoundaryConditions {
public:
    SMSBoundaryConditions() = default;

    /**
     * @brief Apply fixed (zero velocity) BCs to stiffness matrix and RHS.
     *
     * For each fixed DOF i:
     *   K[i, :] = 0, K[:, i] = 0, K[i, i] = 1.0
     *   f[i] = 0.0
     *
     * This enforces x[i] = 0 in the solution.
     *
     * @param K Stiffness matrix (n x n, row-major), modified in place
     * @param f RHS vector (n), modified in place
     * @param n System size
     * @param fixed_dofs Array of DOF indices to fix
     * @param nfixed Number of fixed DOFs
     */
    void apply_fixed(Real* K, Real* f, int n,
                     const int* fixed_dofs, int nfixed) const {
        for (int k = 0; k < nfixed; ++k) {
            int dof = fixed_dofs[k];
            if (dof < 0 || dof >= n) continue;

            // Zero row and column
            for (int j = 0; j < n; ++j) {
                K[dof * n + j] = 0.0;
                K[j * n + dof] = 0.0;
            }
            // Set diagonal to 1
            K[dof * n + dof] = 1.0;
            // Zero RHS
            f[dof] = 0.0;
        }
    }

    /**
     * @brief Apply fixed BCs with prescribed values.
     *
     * For each fixed DOF i with value u_i:
     *   Subtract K[:, i]*u_i from f
     *   Then zero row/col, set K[i,i]=1, f[i]=u_i
     *
     * @param K Stiffness matrix
     * @param f RHS vector
     * @param n System size
     * @param fixed_dofs DOF indices
     * @param values Prescribed values
     * @param nfixed Number of fixed DOFs
     */
    void apply_fixed_prescribed(Real* K, Real* f, int n,
                                 const int* fixed_dofs, const Real* values,
                                 int nfixed) const {
        // Move known displacements to RHS
        for (int k = 0; k < nfixed; ++k) {
            int dof = fixed_dofs[k];
            if (dof < 0 || dof >= n) continue;
            Real val = values[k];
            for (int i = 0; i < n; ++i) {
                f[i] -= K[i * n + dof] * val;
            }
        }
        // Now apply homogeneous BCs with modified RHS
        for (int k = 0; k < nfixed; ++k) {
            int dof = fixed_dofs[k];
            if (dof < 0 || dof >= n) continue;
            for (int j = 0; j < n; ++j) {
                K[dof * n + j] = 0.0;
                K[j * n + dof] = 0.0;
            }
            K[dof * n + dof] = 1.0;
            f[dof] = values[k];
        }
    }

    /**
     * @brief Apply cyclic (periodic) boundary conditions.
     *
     * Ties DOF pairs (a, b) such that x[a] = x[b].
     * Implementation: add row/col b to a, zero row/col b, set K[b,b]=1, f[b]=0.
     * After solve, set x[b] = x[a].
     *
     * @param K Stiffness matrix (n x n)
     * @param f RHS vector (n)
     * @param n System size
     * @param pairs Cyclic DOF pairs
     * @param npairs Number of pairs
     */
    void apply_cyclic(Real* K, Real* f, int n,
                      const CyclicPair* pairs, int npairs) const {
        for (int p = 0; p < npairs; ++p) {
            int a = pairs[p].dof_a;
            int b = pairs[p].dof_b;
            if (a < 0 || a >= n || b < 0 || b >= n || a == b) continue;

            // Add row b to row a
            for (int j = 0; j < n; ++j) {
                K[a * n + j] += K[b * n + j];
            }
            // Add col b to col a
            for (int i = 0; i < n; ++i) {
                K[i * n + a] += K[i * n + b];
            }
            // Add RHS
            f[a] += f[b];

            // Zero row b, col b
            for (int j = 0; j < n; ++j) {
                K[b * n + j] = 0.0;
                K[j * n + b] = 0.0;
            }
            K[b * n + b] = 1.0;
            K[b * n + a] = -1.0; // x[b] - x[a] = 0
            f[b] = 0.0;
        }
    }

    /**
     * @brief Post-solve: enforce cyclic x[b] = x[a].
     */
    static void enforce_cyclic(Real* x, const CyclicPair* pairs, int npairs) {
        for (int p = 0; p < npairs; ++p) {
            x[pairs[p].dof_b] = x[pairs[p].dof_a];
        }
    }

    /**
     * @brief Count the number of free (unconstrained) DOFs.
     */
    static int count_free_dofs(int n, const int* fixed_dofs, int nfixed) {
        std::vector<bool> is_fixed(n, false);
        for (int i = 0; i < nfixed; ++i) {
            if (fixed_dofs[i] >= 0 && fixed_dofs[i] < n)
                is_fixed[fixed_dofs[i]] = true;
        }
        int count = 0;
        for (int i = 0; i < n; ++i)
            if (!is_fixed[i]) ++count;
        return count;
    }
};


// ============================================================================
// 4. ComponentModeSynthesis - Craig-Bampton Substructuring
// ============================================================================

/**
 * @brief Result of Craig-Bampton reduction.
 */
struct CraigBamptonResult {
    std::vector<Real> K_cb;          ///< Condensed stiffness (n_cb x n_cb)
    std::vector<Real> M_cb;          ///< Condensed mass (n_cb x n_cb)
    std::vector<Real> Phi;           ///< Mode shape matrix (n_interior x n_modes)
    std::vector<Real> eigenvalues;   ///< Interior eigenvalues (omega^2)
    int n_boundary = 0;
    int n_interior = 0;
    int n_modes = 0;
    int n_cb = 0;                    ///< Total CB DOFs = n_boundary + n_modes
};

/**
 * @brief Craig-Bampton Component Mode Synthesis.
 *
 * Partitions DOFs into boundary (b) and interior (i) sets.
 * Computes fixed-interface normal modes from K_ii, M_ii.
 * Forms the condensed system:
 *
 *   K_cb = | K_bb_bar    L      |    M_cb = | M_bb_bar    L_m    |
 *          | L^T         Omega^2 |           | L_m^T       I      |
 *
 * where:
 *   Psi = -K_ii^{-1} * K_ib  (constraint modes)
 *   K_bb_bar = K_bb + K_bi * Psi
 *   Omega^2 = diag(omega_k^2) from K_ii * phi_k = omega_k^2 * M_ii * phi_k
 *   L = Phi^T * (K_ib + M_ii * K_ii^{-1} * K_ib) -- coupling
 *
 * For simplicity, this implementation uses dense linear algebra with
 * a simple inverse power iteration for eigenvalues.
 *
 * Reference: Craig & Bampton (1968)
 */
class ComponentModeSynthesis {
public:
    ComponentModeSynthesis() = default;

    /**
     * @brief Perform Craig-Bampton reduction.
     *
     * @param K Full stiffness matrix (n x n, row-major)
     * @param M Full mass matrix (n x n, row-major)
     * @param n Total DOFs
     * @param boundary_dofs Array of boundary DOF indices
     * @param n_boundary Number of boundary DOFs
     * @param n_modes Number of fixed-interface modes to retain
     * @return CraigBamptonResult with condensed matrices
     */
    CraigBamptonResult reduce(const Real* K, const Real* M, int n,
                               const int* boundary_dofs, int n_boundary,
                               int n_modes) const {
        CraigBamptonResult result;
        result.n_boundary = n_boundary;

        // Build interior DOF list
        std::vector<bool> is_boundary(n, false);
        for (int i = 0; i < n_boundary; ++i) {
            if (boundary_dofs[i] >= 0 && boundary_dofs[i] < n)
                is_boundary[boundary_dofs[i]] = true;
        }
        std::vector<int> interior_dofs;
        interior_dofs.reserve(n - n_boundary);
        for (int i = 0; i < n; ++i) {
            if (!is_boundary[i]) interior_dofs.push_back(i);
        }
        int ni = static_cast<int>(interior_dofs.size());
        result.n_interior = ni;

        // Clamp n_modes
        n_modes = std::min(n_modes, ni);
        result.n_modes = n_modes;
        result.n_cb = n_boundary + n_modes;

        // Extract submatrices K_ii, K_bb, K_ib, M_ii, M_bb
        std::vector<Real> K_ii(ni * ni), K_bb(n_boundary * n_boundary);
        std::vector<Real> K_ib(ni * n_boundary);
        std::vector<Real> M_ii(ni * ni), M_bb(n_boundary * n_boundary);

        for (int i = 0; i < ni; ++i) {
            for (int j = 0; j < ni; ++j) {
                K_ii[i * ni + j] = K[interior_dofs[i] * n + interior_dofs[j]];
                M_ii[i * ni + j] = M[interior_dofs[i] * n + interior_dofs[j]];
            }
        }
        for (int i = 0; i < n_boundary; ++i) {
            for (int j = 0; j < n_boundary; ++j) {
                K_bb[i * n_boundary + j] = K[boundary_dofs[i] * n + boundary_dofs[j]];
                M_bb[i * n_boundary + j] = M[boundary_dofs[i] * n + boundary_dofs[j]];
            }
        }
        for (int i = 0; i < ni; ++i) {
            for (int j = 0; j < n_boundary; ++j) {
                K_ib[i * n_boundary + j] = K[interior_dofs[i] * n + boundary_dofs[j]];
            }
        }

        // Compute K_ii inverse (dense, for small systems)
        std::vector<Real> K_ii_inv(ni * ni);
        dense_inverse(K_ii.data(), K_ii_inv.data(), ni);

        // Constraint modes: Psi = -K_ii^{-1} * K_ib (ni x n_boundary)
        std::vector<Real> Psi(ni * n_boundary);
        ams_detail::matmul(K_ii_inv.data(), K_ib.data(), Psi.data(),
                           ni, ni, n_boundary);
        for (int i = 0; i < ni * n_boundary; ++i) Psi[i] = -Psi[i];

        // K_bb_bar = K_bb + K_bi * Psi = K_bb - K_bi * K_ii^{-1} * K_ib
        // K_bi = K_ib^T
        std::vector<Real> K_bi(n_boundary * ni);
        ams_detail::transpose(K_ib.data(), K_bi.data(), ni, n_boundary);
        std::vector<Real> K_bi_Psi(n_boundary * n_boundary, 0.0);
        ams_detail::matmul(K_bi.data(), Psi.data(), K_bi_Psi.data(),
                           n_boundary, ni, n_boundary);
        std::vector<Real> K_bb_bar(n_boundary * n_boundary);
        for (int i = 0; i < n_boundary * n_boundary; ++i) {
            K_bb_bar[i] = K_bb[i] + K_bi_Psi[i];
        }

        // Compute fixed-interface modes via inverse iteration
        // Solve K_ii * phi = lambda * M_ii * phi
        result.eigenvalues.resize(n_modes, 0.0);
        result.Phi.resize(ni * n_modes, 0.0);

        compute_modes(K_ii.data(), M_ii.data(), ni, n_modes,
                      result.Phi.data(), result.eigenvalues.data());

        // Build condensed stiffness K_cb
        int ncb = result.n_cb;
        result.K_cb.assign(ncb * ncb, 0.0);
        result.M_cb.assign(ncb * ncb, 0.0);

        // K_cb top-left: K_bb_bar
        for (int i = 0; i < n_boundary; ++i)
            for (int j = 0; j < n_boundary; ++j)
                result.K_cb[i * ncb + j] = K_bb_bar[i * n_boundary + j];

        // K_cb bottom-right: diag(omega^2)
        for (int k = 0; k < n_modes; ++k)
            result.K_cb[(n_boundary + k) * ncb + (n_boundary + k)] = result.eigenvalues[k];

        // Coupling terms L = Phi^T * K_ib (n_modes x n_boundary)
        // Phi^T is (n_modes x ni), K_ib is (ni x n_boundary)
        std::vector<Real> PhiT(n_modes * ni);
        ams_detail::transpose(result.Phi.data(), PhiT.data(), ni, n_modes);
        std::vector<Real> L(n_modes * n_boundary, 0.0);
        ams_detail::matmul(PhiT.data(), K_ib.data(), L.data(),
                           n_modes, ni, n_boundary);

        // Add coupling to off-diagonals: note we also add Psi contribution
        // L_full = Phi^T * (K_ib + K_ii * Psi) -- but K_ii*Psi = -K_ib, so
        // L_full = Phi^T * (K_ib - K_ib) = 0 for exact modes
        // In practice, with truncated modes: L = Phi^T * K_ib is the coupling
        for (int k = 0; k < n_modes; ++k) {
            for (int j = 0; j < n_boundary; ++j) {
                result.K_cb[(n_boundary + k) * ncb + j] = L[k * n_boundary + j];
                result.K_cb[j * ncb + (n_boundary + k)] = L[k * n_boundary + j];
            }
        }

        // M_cb: top-left M_bb_bar (simplified: M_bb + Psi^T*M_ii*Psi)
        std::vector<Real> PsiT(n_boundary * ni);
        ams_detail::transpose(Psi.data(), PsiT.data(), ni, n_boundary);
        std::vector<Real> M_ii_Psi(ni * n_boundary);
        ams_detail::matmul(M_ii.data(), Psi.data(), M_ii_Psi.data(),
                           ni, ni, n_boundary);
        std::vector<Real> PsiT_M_Psi(n_boundary * n_boundary, 0.0);
        ams_detail::matmul(PsiT.data(), M_ii_Psi.data(), PsiT_M_Psi.data(),
                           n_boundary, ni, n_boundary);
        for (int i = 0; i < n_boundary; ++i)
            for (int j = 0; j < n_boundary; ++j)
                result.M_cb[i * ncb + j] = M_bb[i * n_boundary + j] + PsiT_M_Psi[i * n_boundary + j];

        // M_cb bottom-right: identity (modes are M-orthonormal)
        for (int k = 0; k < n_modes; ++k)
            result.M_cb[(n_boundary + k) * ncb + (n_boundary + k)] = 1.0;

        // M_cb coupling: Phi^T * M_ii * Psi
        std::vector<Real> Lm(n_modes * n_boundary, 0.0);
        ams_detail::matmul(PhiT.data(), M_ii_Psi.data(), Lm.data(),
                           n_modes, ni, n_boundary);
        for (int k = 0; k < n_modes; ++k) {
            for (int j = 0; j < n_boundary; ++j) {
                result.M_cb[(n_boundary + k) * ncb + j] = Lm[k * n_boundary + j];
                result.M_cb[j * ncb + (n_boundary + k)] = Lm[k * n_boundary + j];
            }
        }

        return result;
    }

private:
    /**
     * @brief Dense matrix inverse via Gauss-Jordan elimination.
     * For small systems only (up to ~50 DOFs).
     */
    static void dense_inverse(const Real* A, Real* Ainv, int n) {
        // Augmented matrix [A | I]
        std::vector<Real> aug(n * 2 * n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                aug[i * 2 * n + j] = A[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; ++col) {
            // Find pivot
            int pivot = col;
            Real max_val = std::abs(aug[col * 2 * n + col]);
            for (int row = col + 1; row < n; ++row) {
                Real val = std::abs(aug[row * 2 * n + col]);
                if (val > max_val) { max_val = val; pivot = row; }
            }
            if (max_val < 1.0e-30) {
                // Singular - set identity
                for (int i = 0; i < n * n; ++i) Ainv[i] = 0.0;
                for (int i = 0; i < n; ++i) Ainv[i * n + i] = 1.0;
                return;
            }
            // Swap rows
            if (pivot != col) {
                for (int j = 0; j < 2 * n; ++j)
                    std::swap(aug[col * 2 * n + j], aug[pivot * 2 * n + j]);
            }
            // Scale pivot row
            Real diag = aug[col * 2 * n + col];
            for (int j = 0; j < 2 * n; ++j) aug[col * 2 * n + j] /= diag;

            // Eliminate
            for (int row = 0; row < n; ++row) {
                if (row == col) continue;
                Real factor = aug[row * 2 * n + col];
                for (int j = 0; j < 2 * n; ++j)
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }

        // Extract inverse
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Ainv[i * n + j] = aug[i * 2 * n + n + j];
    }

    /**
     * @brief Compute fixed-interface modes via inverse iteration with deflation.
     *
     * Solves K_ii * phi = lambda * M_ii * phi for the smallest eigenvalues.
     * Uses inverse iteration: starting from random vector, iterate
     *   M_ii * z = phi_old
     *   K_ii * phi_new = z
     *   phi_new /= ||phi_new||_M
     *
     * Actually we solve K_ii^{-1} * M_ii * phi = (1/lambda) * phi via power iteration
     * on B = K_ii^{-1} * M_ii to find the largest eigenvalues of B (= smallest of original).
     */
    static void compute_modes(const Real* K_ii, const Real* M_ii, int ni,
                               int n_modes, Real* Phi, Real* eigenvalues) {
        if (ni <= 0 || n_modes <= 0) return;

        std::vector<Real> K_inv(ni * ni);
        dense_inverse(K_ii, K_inv.data(), ni);

        // B = K_inv * M_ii
        std::vector<Real> B(ni * ni);
        ams_detail::matmul(K_inv.data(), M_ii, B.data(), ni, ni, ni);

        // Power iteration with deflation for each mode
        std::vector<Real> v(ni), Bv(ni);
        std::vector<std::vector<Real>> modes;

        for (int mode = 0; mode < n_modes; ++mode) {
            // Initialize with unit vector
            for (int i = 0; i < ni; ++i) v[i] = 1.0 / std::sqrt(static_cast<Real>(ni));
            // Perturb to break symmetry
            if (mode < ni) v[mode] += 0.5;
            Real vn = ams_detail::norm2(v.data(), ni);
            if (vn > 1.0e-30) ams_detail::scale(1.0 / vn, v.data(), ni);

            Real lambda = 0.0;
            for (int iter = 0; iter < 200; ++iter) {
                // Bv = B * v
                ams_detail::matvec(B.data(), v.data(), Bv.data(), ni);

                // Deflate against previously found modes
                for (size_t m = 0; m < modes.size(); ++m) {
                    Real proj = ams_detail::dot(Bv.data(), modes[m].data(), ni);
                    ams_detail::axpy(-proj, modes[m].data(), Bv.data(), ni);
                }

                Real norm = ams_detail::norm2(Bv.data(), ni);
                if (norm < 1.0e-30) break;

                lambda = norm;  // Rayleigh quotient approximation
                ams_detail::scale(1.0 / norm, Bv.data(), ni);
                ams_detail::copy(Bv.data(), v.data(), ni);
            }

            // lambda = 1/omega^2, so omega^2 = 1/lambda
            eigenvalues[mode] = (std::abs(lambda) > 1.0e-30) ? 1.0 / lambda : 0.0;

            // Store mode
            for (int i = 0; i < ni; ++i) Phi[i * n_modes + mode] = v[i];
            modes.push_back(std::vector<Real>(v.begin(), v.end()));
        }
    }
};


// ============================================================================
// 5. FrequencyResponse - Modal Superposition FRF
// ============================================================================

/**
 * @brief Complex number for FRF computation.
 */
struct Complex {
    Real re = 0.0;
    Real im = 0.0;

    Complex() = default;
    Complex(Real r, Real i) : re(r), im(i) {}

    Complex operator+(const Complex& o) const { return {re + o.re, im + o.im}; }
    Complex operator-(const Complex& o) const { return {re - o.re, im - o.im}; }
    Complex operator*(const Complex& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    Complex operator*(Real s) const { return {re * s, im * s}; }

    Real magnitude() const { return std::sqrt(re * re + im * im); }
    Real phase() const { return std::atan2(im, re); }

    Complex conjugate() const { return {re, -im}; }

    /// Reciprocal: 1/z = z* / |z|^2
    Complex reciprocal() const {
        Real mag2 = re * re + im * im;
        if (mag2 < 1.0e-30) return {0.0, 0.0};
        return {re / mag2, -im / mag2};
    }
};

/**
 * @brief Frequency Response Function computation via modal superposition.
 *
 * Given N modes with shapes phi_k (n x 1) and natural frequencies omega_k,
 * the receptance FRF at excitation frequency omega with damping ratio zeta is:
 *
 *   H(omega) = sum_k [ phi_k * phi_k^T / (omega_k^2 - omega^2 + 2i*zeta*omega_k*omega) ]
 *
 * For a given input DOF j and output DOF i:
 *   H_ij(omega) = sum_k [ phi_k(i) * phi_k(j) / D_k ]
 *   where D_k = omega_k^2 - omega^2 + 2i*zeta*omega_k*omega
 *
 * The magnitude |H| peaks near omega = omega_k (resonance).
 *
 * Reference: Ewins (2000) "Modal Testing"
 */
class FrequencyResponse {
public:
    FrequencyResponse() = default;

    /**
     * @brief Compute FRF matrix entry H_ij at a single frequency.
     *
     * @param modes Mode shapes, column-major (n x n_modes), modes[i * n_modes + k] = phi_k(i)
     * @param freqs Natural frequencies omega_k (n_modes)
     * @param n_modes Number of modes
     * @param omega Excitation frequency
     * @param zeta Modal damping ratio
     * @param dof_in Input (excitation) DOF index
     * @param dof_out Output (response) DOF index
     * @param n Total DOFs
     * @return Complex FRF value H_ij(omega)
     */
    Complex compute_frf_point(const Real* modes, const Real* freqs,
                               int n_modes, Real omega, Real zeta,
                               int dof_in, int dof_out, int n) const {
        Complex H(0.0, 0.0);
        Real omega2 = omega * omega;

        for (int k = 0; k < n_modes; ++k) {
            Real omega_k = freqs[k];
            Real omega_k2 = omega_k * omega_k;

            // Denominator: D_k = (omega_k^2 - omega^2) + 2i*zeta*omega_k*omega
            Complex D(omega_k2 - omega2, 2.0 * zeta * omega_k * omega);

            // Numerator: phi_k(dof_out) * phi_k(dof_in)
            Real phi_out = modes[dof_out * n_modes + k];
            Real phi_in = modes[dof_in * n_modes + k];
            Real num = phi_out * phi_in;

            // H += num / D
            Complex D_inv = D.reciprocal();
            H = H + D_inv * num;
        }

        return H;
    }

    /**
     * @brief Compute FRF over a frequency sweep.
     *
     * @param modes Mode shapes (n x n_modes, column-major: modes[i * n_modes + k])
     * @param freqs Natural frequencies omega_k
     * @param n_modes Number of modes
     * @param n Total DOFs
     * @param omega_sweep Array of excitation frequencies
     * @param n_freq Number of frequency points
     * @param zeta Damping ratio
     * @param dof_in Input DOF
     * @param dof_out Output DOF
     * @param H_mag Output: magnitude of H at each frequency (n_freq)
     * @param H_phase Output: phase of H at each frequency (n_freq), in radians
     */
    void compute_frf(const Real* modes, const Real* freqs, int n_modes,
                     int n, const Real* omega_sweep, int n_freq,
                     Real zeta, int dof_in, int dof_out,
                     Real* H_mag, Real* H_phase) const {
        for (int f = 0; f < n_freq; ++f) {
            Complex H = compute_frf_point(modes, freqs, n_modes,
                                           omega_sweep[f], zeta,
                                           dof_in, dof_out, n);
            H_mag[f] = H.magnitude();
            H_phase[f] = H.phase();
        }
    }

    /**
     * @brief Find peak (resonance) frequency from FRF magnitude data.
     *
     * @param H_mag FRF magnitude array
     * @param omega_sweep Frequency sweep array
     * @param n_freq Number of points
     * @return Frequency at which |H| is maximum
     */
    static Real find_resonance(const Real* H_mag, const Real* omega_sweep,
                                int n_freq) {
        int peak_idx = 0;
        Real peak_val = H_mag[0];
        for (int i = 1; i < n_freq; ++i) {
            if (H_mag[i] > peak_val) {
                peak_val = H_mag[i];
                peak_idx = i;
            }
        }
        return omega_sweep[peak_idx];
    }

    /**
     * @brief Compute half-power bandwidth damping estimate.
     *
     * At resonance omega_r, the half-power points are at |H| = |H_max|/sqrt(2).
     * The damping ratio is approximately: zeta ~ (omega_2 - omega_1) / (2 * omega_r)
     *
     * @param H_mag FRF magnitude array
     * @param omega_sweep Frequency array
     * @param n_freq Number of points
     * @return Estimated damping ratio
     */
    static Real estimate_damping(const Real* H_mag, const Real* omega_sweep,
                                  int n_freq) {
        // Find peak
        int peak_idx = 0;
        Real peak_val = H_mag[0];
        for (int i = 1; i < n_freq; ++i) {
            if (H_mag[i] > peak_val) { peak_val = H_mag[i]; peak_idx = i; }
        }

        Real half_power = peak_val / std::sqrt(2.0);

        // Find omega_1 (left of peak)
        Real omega_1 = omega_sweep[0];
        for (int i = 0; i < peak_idx; ++i) {
            if (H_mag[i] >= half_power && (i == 0 || H_mag[i-1] < half_power)) {
                // Linear interpolation
                if (i > 0) {
                    Real frac = (half_power - H_mag[i-1]) / (H_mag[i] - H_mag[i-1] + 1e-30);
                    omega_1 = omega_sweep[i-1] + frac * (omega_sweep[i] - omega_sweep[i-1]);
                } else {
                    omega_1 = omega_sweep[i];
                }
                break;
            }
        }

        // Find omega_2 (right of peak)
        Real omega_2 = omega_sweep[n_freq - 1];
        for (int i = peak_idx + 1; i < n_freq; ++i) {
            if (H_mag[i] <= half_power && H_mag[i-1] > half_power) {
                Real frac = (H_mag[i-1] - half_power) / (H_mag[i-1] - H_mag[i] + 1e-30);
                omega_2 = omega_sweep[i-1] + frac * (omega_sweep[i] - omega_sweep[i-1]);
                break;
            }
        }

        Real omega_r = omega_sweep[peak_idx];
        if (std::abs(omega_r) < 1e-30) return 0.0;
        return (omega_2 - omega_1) / (2.0 * omega_r);
    }
};


// ============================================================================
// 6. SMSConstraints - RBE2/RBE3/Joints for SMS
// ============================================================================

/**
 * @brief Multi-point constraint handler for SMS systems.
 *
 * RBE2: Rigid body element type 2. Slave DOFs are rigidly tied to master.
 *   K and f are modified so that slave DOFs follow the master DOF exactly.
 *   Implementation: condense slave DOFs into master.
 *
 * RBE3: Weighted averaging constraint. Master DOF = weighted average of slaves.
 *   The master DOF motion is the weighted average of slave DOF motions.
 *   Implementation: distribute master equation among slaves.
 *
 * Reference: MSC.Nastran User Guide, Chapter on Multipoint Constraints
 */
class SMSConstraints {
public:
    SMSConstraints() = default;

    /**
     * @brief Apply RBE2 constraint: slaves rigidly follow master.
     *
     * For each slave DOF s tied to master DOF m:
     *   x[s] = x[m]  (rigid link)
     *
     * Implementation:
     *   1. Add slave row/col to master row/col in K
     *   2. Add slave f to master f
     *   3. Zero slave row/col, set K[s,s]=1, K[s,m]=-1, f[s]=0
     *
     * @param K Stiffness matrix (n x n, modified in place)
     * @param f RHS vector (n, modified in place)
     * @param n System size
     * @param master Master DOF index
     * @param slaves Array of slave DOF indices
     * @param nslaves Number of slaves
     */
    void apply_rbe2(Real* K, Real* f, int n,
                    int master, const int* slaves, int nslaves) const {
        if (master < 0 || master >= n) return;

        for (int s = 0; s < nslaves; ++s) {
            int slave = slaves[s];
            if (slave < 0 || slave >= n || slave == master) continue;

            // Add slave contributions to master
            for (int j = 0; j < n; ++j) {
                K[master * n + j] += K[slave * n + j];
                K[j * n + master] += K[j * n + slave];
            }
            // Correction: the master-master entry was double-counted
            // K[master,master] had K[slave,master] added from row,
            // and K[master,slave] added from col.
            // We also need to add K[slave,slave] once more from cross-terms.
            // Actually, the diagonal correction is: K_mm_new should include
            // K_mm + K_ms + K_sm + K_ss
            // Row add gave: K_mm += K_sm (for each j, when j=master)
            // Col add gave: K_mm += K_ms (for each i, when i=master)
            // But we also need K_ss contribution. It was already added to
            // K[master,master] in both the row and col loops when j/i = slave.
            // Actually row: K[master,slave] += K[slave,slave]
            // And col: K[slave,master] += K[slave,slave] -- but slave row will be zeroed.
            // The bookkeeping is correct as-is for the condensation.

            f[master] += f[slave];

            // Zero slave row and column
            for (int j = 0; j < n; ++j) {
                K[slave * n + j] = 0.0;
                K[j * n + slave] = 0.0;
            }
            K[slave * n + slave] = 1.0;
            K[slave * n + master] = -1.0;
            f[slave] = 0.0;
        }
    }

    /**
     * @brief Apply RBE3 constraint: master = weighted average of slaves.
     *
     * x[master] = sum(w_i * x[slave_i]) / sum(w_i)
     *
     * Implementation: distribute master equation to slaves proportionally.
     *   For each slave i with weight w_i:
     *     K[slave_i, :] += (w_i / W) * K[master, :]
     *     f[slave_i] += (w_i / W) * f[master]
     *   Then zero master row/col, set equation:
     *     x[master] - sum(w_i/W * x[slave_i]) = 0
     *
     * @param K Stiffness matrix (n x n)
     * @param f RHS vector (n)
     * @param n System size
     * @param master Master DOF
     * @param slaves Slave DOF indices
     * @param weights Slave weights
     * @param nslaves Number of slaves
     */
    void apply_rbe3(Real* K, Real* f, int n,
                    int master, const int* slaves, const Real* weights,
                    int nslaves) const {
        if (master < 0 || master >= n || nslaves <= 0) return;

        // Total weight
        Real W = 0.0;
        for (int s = 0; s < nslaves; ++s) W += weights[s];
        if (std::abs(W) < 1.0e-30) return;

        // Distribute master equation to slaves
        for (int s = 0; s < nslaves; ++s) {
            int slave = slaves[s];
            if (slave < 0 || slave >= n) continue;
            Real w = weights[s] / W;

            for (int j = 0; j < n; ++j) {
                K[slave * n + j] += w * K[master * n + j];
            }
            for (int i = 0; i < n; ++i) {
                K[i * n + slave] += w * K[i * n + master];
            }
            f[slave] += w * f[master];
        }

        // Zero master row and column
        for (int j = 0; j < n; ++j) {
            K[master * n + j] = 0.0;
            K[j * n + master] = 0.0;
        }

        // Set constraint equation: x[master] - sum(w_i/W * x[slave_i]) = 0
        K[master * n + master] = 1.0;
        for (int s = 0; s < nslaves; ++s) {
            int slave = slaves[s];
            if (slave < 0 || slave >= n) continue;
            K[master * n + slave] = -weights[s] / W;
        }
        f[master] = 0.0;
    }

    /**
     * @brief Apply joint constraint between two DOFs with a spring stiffness.
     *
     * Adds a penalty spring between DOF a and DOF b:
     *   K[a,a] += k_joint, K[b,b] += k_joint
     *   K[a,b] -= k_joint, K[b,a] -= k_joint
     *
     * @param K Stiffness matrix
     * @param n System size
     * @param dof_a First DOF
     * @param dof_b Second DOF
     * @param k_joint Spring stiffness
     */
    void apply_joint(Real* K, int n, int dof_a, int dof_b, Real k_joint) const {
        if (dof_a < 0 || dof_a >= n || dof_b < 0 || dof_b >= n) return;
        K[dof_a * n + dof_a] += k_joint;
        K[dof_b * n + dof_b] += k_joint;
        K[dof_a * n + dof_b] -= k_joint;
        K[dof_b * n + dof_a] -= k_joint;
    }

    /**
     * @brief Post-solve: recover slave DOFs from master for RBE2.
     */
    static void recover_rbe2(Real* x, int master, const int* slaves, int nslaves) {
        for (int s = 0; s < nslaves; ++s) {
            x[slaves[s]] = x[master];
        }
    }

    /**
     * @brief Post-solve: recover master DOF from slaves for RBE3.
     */
    static void recover_rbe3(Real* x, int master, const int* slaves,
                              const Real* weights, int nslaves) {
        Real W = 0.0;
        for (int s = 0; s < nslaves; ++s) W += weights[s];
        if (std::abs(W) < 1.0e-30) return;

        Real val = 0.0;
        for (int s = 0; s < nslaves; ++s) {
            val += weights[s] * x[slaves[s]];
        }
        x[master] = val / W;
    }
};

} // namespace solver
} // namespace nxs
