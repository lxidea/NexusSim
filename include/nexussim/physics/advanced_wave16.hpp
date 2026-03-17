#pragma once

/**
 * @file advanced_wave16.hpp
 * @brief Wave 16: Advanced Capabilities for NexusSim
 *
 * Sub-modules:
 * - 16a: Modal Analysis (Lanczos eigensolver)
 * - 16b: XFEM (Extended FEM with level-set cracks)
 * - 16c: Blast Loading (CONWEP empirical model)
 * - 16d: Airbag Simulation (gas inflation model)
 * - 16e: Seatbelt Dynamics (belt element + retractor)
 * - 16f: Advanced ALE (Eulerian solver, cut-cell, turbulence)
 * - 16g: Adaptive Mesh Refinement (ZZ error estimator, h-refinement)
 * - 16h: Draping Analysis (kinematic/geodesic fiber mapping)
 */

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include <stdexcept>
#include <cstring>

namespace nxs {
namespace advanced {

using Real = double;
using Index = std::size_t;

// ============================================================================
// 16a: Modal Analysis — Lanczos Eigensolver
// ============================================================================

/**
 * @brief Sparse symmetric matrix stored in CSR format.
 *
 * Used to represent stiffness (K) and mass (M) matrices for modal analysis.
 */
struct SparseMatrix {
    Index n = 0;                     ///< Matrix dimension (n x n)
    std::vector<Index> row_ptr;      ///< Row pointers (size n+1)
    std::vector<Index> col_idx;      ///< Column indices
    std::vector<Real> values;        ///< Non-zero values

    SparseMatrix() = default;

    explicit SparseMatrix(Index dim) : n(dim), row_ptr(dim + 1, 0) {}

    /// Build from dense matrix (for testing / small problems)
    void from_dense(const std::vector<Real>& dense, Index dim) {
        n = dim;
        row_ptr.resize(n + 1, 0);
        col_idx.clear();
        values.clear();
        for (Index i = 0; i < n; ++i) {
            row_ptr[i] = col_idx.size();
            for (Index j = 0; j < n; ++j) {
                Real v = dense[i * n + j];
                if (std::abs(v) > 1.0e-30) {
                    col_idx.push_back(j);
                    values.push_back(v);
                }
            }
        }
        row_ptr[n] = col_idx.size();
    }

    /// Matrix-vector product: y = A * x
    void matvec(const std::vector<Real>& x, std::vector<Real>& y) const {
        y.assign(n, 0.0);
        for (Index i = 0; i < n; ++i) {
            Real sum = 0.0;
            for (Index k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                sum += values[k] * x[col_idx[k]];
            }
            y[i] = sum;
        }
    }
};

/**
 * @brief Result of an eigenvalue computation.
 */
struct EigenResult {
    std::vector<Real> eigenvalues;              ///< Sorted ascending
    std::vector<std::vector<Real>> eigenvectors; ///< Corresponding mode shapes
    int converged_modes = 0;
};

/**
 * @brief Lanczos eigensolver for generalized eigenvalue problem K*x = lambda*M*x.
 *
 * Uses power iteration with deflation and Gram-Schmidt orthogonalization
 * to extract the lowest N eigenpairs. Optionally supports shift-invert
 * mode for interior eigenvalues.
 *
 * Algorithm:
 *   For shift-invert with shift sigma:
 *     Solve (K - sigma*M)^{-1} * M * x = mu * x
 *     where lambda = sigma + 1/mu
 *   For standard mode:
 *     Solve M^{-1} * K * x = lambda * x via inverse iteration on K
 */
class LanczosEigensolver {
public:
    LanczosEigensolver() = default;

    /// Set maximum iterations for power iteration per mode
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }

    /// Set convergence tolerance
    void set_tolerance(Real tol) { tol_ = tol; }

    /// Enable shift-invert with given shift (for interior eigenvalues)
    void set_shift(Real sigma) { sigma_ = sigma; use_shift_ = true; }

    /**
     * @brief Solve K*x = lambda*M*x for the lowest num_modes eigenpairs.
     *
     * Uses inverse iteration with M-orthogonal deflation:
     *   1. For each mode, start with random initial vector
     *   2. Iterate: y = K^{-1} * M * x (using CG solve)
     *   3. Deflate against previously found modes
     *   4. Normalize and check convergence
     *   5. Compute Rayleigh quotient lambda = (x^T K x) / (x^T M x)
     *
     * @param K Stiffness matrix (sparse symmetric positive semi-definite)
     * @param M Mass matrix (sparse symmetric positive definite)
     * @param num_modes Number of eigenvalues/vectors to compute
     * @return EigenResult with sorted eigenvalues and eigenvectors
     */
    EigenResult solve(const SparseMatrix& K, const SparseMatrix& M, int num_modes) {
        Index n = K.n;
        EigenResult result;

        // Effective operator matrix for inverse iteration
        // We solve K * y = M * x iteratively (CG), then the largest eigenvalue
        // of K^{-1}*M corresponds to the smallest eigenvalue of K*x = lambda*M*x

        SparseMatrix Keff;
        if (use_shift_) {
            // Keff = K - sigma * M
            Keff = shift_matrix(K, M, sigma_);
        } else {
            Keff = K;
        }

        std::vector<std::vector<Real>> found_vectors;

        for (int mode = 0; mode < num_modes; ++mode) {
            // Initial vector: seeded deterministically
            std::vector<Real> x(n);
            for (Index i = 0; i < n; ++i) {
                x[i] = 1.0 + 0.1 * std::sin(static_cast<Real>((mode + 1) * (i + 1)));
            }

            // Deflate initial vector against found modes
            deflate(x, found_vectors, M);
            Real norm = m_norm(x, M);
            if (norm < 1.0e-30) {
                // Degenerate: try different seed
                for (Index i = 0; i < n; ++i) {
                    x[i] = std::cos(static_cast<Real>((mode + 7) * (i + 3)));
                }
                deflate(x, found_vectors, M);
                norm = m_norm(x, M);
            }
            scale(x, 1.0 / norm);

            Real lambda_old = 0.0;
            bool converged = false;

            for (int iter = 0; iter < max_iter_; ++iter) {
                // y = M * x
                std::vector<Real> Mx;
                M.matvec(x, Mx);

                // Solve Keff * z = Mx via CG
                std::vector<Real> z = cg_solve(Keff, Mx, 200, 1.0e-12);

                // Deflate
                deflate(z, found_vectors, M);

                // Normalize
                norm = m_norm(z, M);
                if (norm < 1.0e-30) break;
                scale(z, 1.0 / norm);

                // Rayleigh quotient on original K: lambda = (z^T K z) / (z^T M z)
                std::vector<Real> Kz;
                K.matvec(z, Kz);
                Real ztKz = dot(z, Kz);
                std::vector<Real> Mz;
                M.matvec(z, Mz);
                Real ztMz = dot(z, Mz);
                Real lambda = (std::abs(ztMz) > 1.0e-30) ? ztKz / ztMz : 0.0;

                x = z;

                if (iter > 0 && std::abs(lambda - lambda_old) <
                    tol_ * (1.0 + std::abs(lambda))) {
                    converged = true;
                    lambda_old = lambda;
                    break;
                }
                lambda_old = lambda;
            }

            result.eigenvalues.push_back(lambda_old);
            result.eigenvectors.push_back(x);
            found_vectors.push_back(x);
            if (converged) result.converged_modes++;
        }

        // Sort by eigenvalue ascending
        std::vector<int> idx(result.eigenvalues.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return result.eigenvalues[a] < result.eigenvalues[b];
        });

        EigenResult sorted;
        sorted.converged_modes = result.converged_modes;
        for (int i : idx) {
            sorted.eigenvalues.push_back(result.eigenvalues[i]);
            sorted.eigenvectors.push_back(result.eigenvectors[i]);
        }

        return sorted;
    }

    /**
     * @brief Compute natural frequencies (Hz) from eigenvalues.
     * omega_i = sqrt(lambda_i), freq_i = omega_i / (2*pi)
     */
    static std::vector<Real> eigenvalues_to_frequencies(const std::vector<Real>& eigenvalues) {
        std::vector<Real> freqs;
        freqs.reserve(eigenvalues.size());
        for (Real lam : eigenvalues) {
            Real omega = (lam > 0.0) ? std::sqrt(lam) : 0.0;
            freqs.push_back(omega / (2.0 * M_PI));
        }
        return freqs;
    }

private:
    int max_iter_ = 500;
    Real tol_ = 1.0e-8;
    Real sigma_ = 0.0;
    bool use_shift_ = false;

    static Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
        Real s = 0.0;
        for (Index i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    static void scale(std::vector<Real>& v, Real s) {
        for (auto& x : v) x *= s;
    }

    /// M-norm: sqrt(x^T M x)
    Real m_norm(const std::vector<Real>& x, const SparseMatrix& M) const {
        std::vector<Real> Mx;
        M.matvec(x, Mx);
        Real d = dot(x, Mx);
        return (d > 0.0) ? std::sqrt(d) : 0.0;
    }

    /// Deflate x against found eigenvectors (M-orthogonalization)
    void deflate(std::vector<Real>& x, const std::vector<std::vector<Real>>& vecs,
                 const SparseMatrix& M) const {
        for (const auto& v : vecs) {
            std::vector<Real> Mv;
            M.matvec(v, Mv);
            Real coeff = dot(x, Mv);
            for (Index i = 0; i < x.size(); ++i) {
                x[i] -= coeff * v[i];
            }
        }
    }

    /// Direct dense solver (Gaussian elimination with partial pivoting) for small systems
    std::vector<Real> direct_solve(const SparseMatrix& A, const std::vector<Real>& b) const {
        Index n = A.n;
        // Build dense augmented matrix
        std::vector<Real> aug(n * (n + 1), 0.0);
        for (Index i = 0; i < n; ++i) {
            for (Index k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                aug[i * (n + 1) + A.col_idx[k]] = A.values[k];
            }
            aug[i * (n + 1) + n] = b[i];
        }
        // Forward elimination with partial pivoting
        for (Index col = 0; col < n; ++col) {
            // Find pivot
            Index pivot = col;
            Real max_val = std::abs(aug[col * (n + 1) + col]);
            for (Index row = col + 1; row < n; ++row) {
                Real val = std::abs(aug[row * (n + 1) + col]);
                if (val > max_val) { max_val = val; pivot = row; }
            }
            if (max_val < 1.0e-30) continue;
            // Swap rows
            if (pivot != col) {
                for (Index j = 0; j <= n; ++j)
                    std::swap(aug[col * (n + 1) + j], aug[pivot * (n + 1) + j]);
            }
            // Eliminate
            for (Index row = col + 1; row < n; ++row) {
                Real factor = aug[row * (n + 1) + col] / aug[col * (n + 1) + col];
                for (Index j = col; j <= n; ++j) {
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
                }
            }
        }
        // Back substitution
        std::vector<Real> x(n, 0.0);
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            Real s = aug[i * (n + 1) + n];
            for (Index j = i + 1; j < n; ++j) {
                s -= aug[i * (n + 1) + j] * x[j];
            }
            Real diag = aug[i * (n + 1) + i];
            x[i] = (std::abs(diag) > 1.0e-30) ? s / diag : 0.0;
        }
        return x;
    }

    /// CG solver: solve A*x = b (for larger systems)
    std::vector<Real> cg_solve(const SparseMatrix& A, const std::vector<Real>& b,
                               int max_cg, Real cg_tol) const {
        // Use direct solver for small systems (more robust)
        if (A.n <= 64) return direct_solve(A, b);

        Index n = A.n;
        std::vector<Real> x(n, 0.0);
        std::vector<Real> r = b;
        std::vector<Real> p = r;
        Real rsold = dot(r, r);
        if (rsold < cg_tol * cg_tol) return x;

        for (int i = 0; i < max_cg; ++i) {
            std::vector<Real> Ap;
            A.matvec(p, Ap);
            Real pAp = dot(p, Ap);
            if (std::abs(pAp) < 1.0e-30) break;
            Real alpha = rsold / pAp;
            for (Index j = 0; j < n; ++j) {
                x[j] += alpha * p[j];
                r[j] -= alpha * Ap[j];
            }
            Real rsnew = dot(r, r);
            if (std::sqrt(rsnew) < cg_tol) break;
            Real beta = rsnew / rsold;
            for (Index j = 0; j < n; ++j) {
                p[j] = r[j] + beta * p[j];
            }
            rsold = rsnew;
        }
        return x;
    }

    /// Compute Keff = K - sigma * M
    SparseMatrix shift_matrix(const SparseMatrix& K, const SparseMatrix& M, Real sigma) const {
        // Convert both to dense, combine, convert back (for small-to-moderate systems)
        Index n = K.n;
        std::vector<Real> dense(n * n, 0.0);
        // Add K
        for (Index i = 0; i < n; ++i) {
            for (Index k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k) {
                dense[i * n + K.col_idx[k]] += K.values[k];
            }
        }
        // Subtract sigma * M
        for (Index i = 0; i < n; ++i) {
            for (Index k = M.row_ptr[i]; k < M.row_ptr[i + 1]; ++k) {
                dense[i * n + M.col_idx[k]] -= sigma * M.values[k];
            }
        }
        SparseMatrix result;
        result.from_dense(dense, n);
        return result;
    }
};

// ============================================================================
// 16b: XFEM — Extended Finite Element Method
// ============================================================================

/**
 * @brief Level set representation of a crack.
 *
 * Uses two level set fields:
 * - phi: signed distance to crack surface (normal direction)
 * - psi: signed distance to crack front (tangential direction)
 *
 * Crack surface where phi=0, crack front where phi=0 AND psi=0.
 */
class LevelSetCrack {
public:
    LevelSetCrack() = default;

    /// Initialize with mesh node count
    void initialize(Index num_nodes) {
        num_nodes_ = num_nodes;
        phi_.assign(num_nodes, 1.0);   // All nodes above crack by default
        psi_.assign(num_nodes, 1.0);
        crack_tip_ = {0.0, 0.0, 0.0};
    }

    /// Set level set values for a node
    void set_phi(Index node, Real val) { phi_[node] = val; }
    void set_psi(Index node, Real val) { psi_[node] = val; }

    Real get_phi(Index node) const { return phi_[node]; }
    Real get_psi(Index node) const { return psi_[node]; }

    /// Set the crack tip location
    void set_crack_tip(Real x, Real y, Real z = 0.0) {
        crack_tip_ = {x, y, z};
    }

    std::array<Real, 3> crack_tip() const { return crack_tip_; }

    /**
     * @brief Initialize a straight crack from (x0,y0) to (x1,y1).
     *
     * Phi is the signed distance to the crack line.
     * Psi is signed distance along the crack direction, positive ahead of tip.
     */
    void initialize_straight_crack(const std::vector<std::array<Real, 3>>& node_coords,
                                   Real x0, Real y0, Real x1, Real y1) {
        num_nodes_ = node_coords.size();
        phi_.resize(num_nodes_);
        psi_.resize(num_nodes_);

        Real dx = x1 - x0, dy = y1 - y0;
        Real len = std::sqrt(dx * dx + dy * dy);
        if (len < 1.0e-30) return;

        // Crack direction and normal
        Real tx = dx / len, ty = dy / len;
        Real nx = -ty, ny = tx;

        crack_tip_ = {x1, y1, 0.0};

        for (Index i = 0; i < num_nodes_; ++i) {
            Real px = node_coords[i][0] - x0;
            Real py = node_coords[i][1] - y0;
            phi_[i] = px * nx + py * ny;      // Distance to crack line
            psi_[i] = px * tx + py * ty - len; // Distance ahead of tip (negative = behind tip = on crack)
        }
    }

    /**
     * @brief Check if an element is cut by the crack.
     *
     * An element is cut if phi changes sign across its nodes AND
     * at least one node has psi <= 0 (is behind the crack tip).
     */
    bool is_cut(const std::vector<Index>& element_nodes) const {
        if (element_nodes.empty()) return false;

        bool has_pos_phi = false, has_neg_phi = false;
        bool has_behind_tip = false;

        for (Index nd : element_nodes) {
            if (nd >= num_nodes_) continue;
            if (phi_[nd] > 0.0) has_pos_phi = true;
            if (phi_[nd] < 0.0) has_neg_phi = true;
            if (phi_[nd] == 0.0) { has_pos_phi = true; has_neg_phi = true; }
            if (psi_[nd] <= 0.0) has_behind_tip = true;
        }

        return has_pos_phi && has_neg_phi && has_behind_tip;
    }

    /**
     * @brief Check if an element contains the crack tip.
     * Tip is present where both phi and psi change sign.
     */
    bool has_tip(const std::vector<Index>& element_nodes) const {
        if (element_nodes.empty()) return false;
        bool phi_pos = false, phi_neg = false;
        bool psi_pos = false, psi_neg = false;
        for (Index nd : element_nodes) {
            if (nd >= num_nodes_) continue;
            if (phi_[nd] >= 0.0) phi_pos = true;
            if (phi_[nd] <= 0.0) phi_neg = true;
            if (psi_[nd] >= 0.0) psi_pos = true;
            if (psi_[nd] <= 0.0) psi_neg = true;
        }
        return phi_pos && phi_neg && psi_pos && psi_neg;
    }

    Index num_nodes() const { return num_nodes_; }

private:
    Index num_nodes_ = 0;
    std::vector<Real> phi_;
    std::vector<Real> psi_;
    std::array<Real, 3> crack_tip_ = {0.0, 0.0, 0.0};
};

/**
 * @brief XFEM enrichment functions for cracked elements.
 *
 * Provides:
 * - Heaviside enrichment H(phi) for fully cut elements
 * - Crack-tip enrichment {sqrt(r)*sin(t/2), sqrt(r)*cos(t/2),
 *   sqrt(r)*sin(t/2)*sin(t), sqrt(r)*cos(t/2)*sin(t)} for tip elements
 */
class XFEMEnrichment {
public:
    XFEMEnrichment() = default;

    /**
     * @brief Heaviside enrichment function.
     * H(phi) = +1 if phi > 0, -1 if phi < 0, 0 if phi == 0
     */
    static Real heaviside(Real phi) {
        if (phi > 0.0) return 1.0;
        if (phi < 0.0) return -1.0;
        return 0.0;
    }

    /**
     * @brief Crack tip enrichment functions (2D).
     *
     * Given local crack-tip polar coordinates (r, theta):
     *   F1 = sqrt(r) * sin(theta/2)
     *   F2 = sqrt(r) * cos(theta/2)
     *   F3 = sqrt(r) * sin(theta/2) * sin(theta)
     *   F4 = sqrt(r) * cos(theta/2) * sin(theta)
     *
     * These span the asymptotic crack-tip displacement field.
     */
    static std::array<Real, 4> crack_tip_functions(Real r, Real theta) {
        Real sqr = std::sqrt(std::max(r, 0.0));
        Real ht = theta * 0.5;
        Real st = std::sin(theta);
        return {
            sqr * std::sin(ht),
            sqr * std::cos(ht),
            sqr * std::sin(ht) * st,
            sqr * std::cos(ht) * st
        };
    }

    /**
     * @brief Compute crack-tip polar coordinates from Cartesian position
     *        relative to crack tip, given the crack direction.
     *
     * @param x,y Point coordinates
     * @param tip_x, tip_y Crack tip coordinates
     * @param crack_angle Crack propagation angle (radians from x-axis)
     * @return {r, theta}
     */
    static std::array<Real, 2> crack_tip_polar(Real x, Real y,
                                                Real tip_x, Real tip_y,
                                                Real crack_angle) {
        Real dx = x - tip_x;
        Real dy = y - tip_y;
        // Rotate to crack-local frame
        Real ca = std::cos(crack_angle), sa = std::sin(crack_angle);
        Real lx = dx * ca + dy * sa;
        Real ly = -dx * sa + dy * ca;
        Real r = std::sqrt(lx * lx + ly * ly);
        Real theta = std::atan2(ly, lx);
        return {r, theta};
    }

    /**
     * @brief Compute enriched shape function value for a split element.
     *
     * N_enriched_I(x) = N_I(x) * [H(phi(x)) - H(phi_I)]
     *
     * @param N_I Standard shape function value at point
     * @param phi_at_point Level set value at evaluation point
     * @param phi_at_node Level set value at the enriched node
     * @return Enriched shape function contribution
     */
    static Real enriched_shape(Real N_I, Real phi_at_point, Real phi_at_node) {
        return N_I * (heaviside(phi_at_point) - heaviside(phi_at_node));
    }

    /**
     * @brief Number of additional DOFs per enriched node.
     * - Heaviside enrichment: ndim extra DOFs
     * - Tip enrichment: 4*ndim extra DOFs (4 branch functions x ndim)
     */
    static int heaviside_extra_dofs(int ndim) { return ndim; }
    static int tip_extra_dofs(int ndim) { return 4 * ndim; }
};

/**
 * @brief Crack propagation manager.
 *
 * Determines crack growth direction and increment using:
 * - Maximum hoop stress criterion (Erdogan-Sih)
 * - J-integral for energy release rate
 */
class CrackPropagation {
public:
    CrackPropagation() = default;

    /// Set material properties
    void set_material(Real E, Real nu) {
        E_ = E;
        nu_ = nu;
        // Plane strain fracture toughness default
        KIc_ = 1.0e30; // Effectively infinite until set
    }

    /// Set fracture toughness
    void set_fracture_toughness(Real KIc) { KIc_ = KIc; }

    /// Set crack growth increment
    void set_growth_increment(Real da) { da_ = da; }

    /**
     * @brief Maximum hoop stress criterion (Erdogan-Sih 1963).
     *
     * Given mode I and II SIFs, the crack propagation angle theta_c is:
     *   theta_c = 2 * atan( (KI - sqrt(KI^2 + 8*KII^2)) / (4*KII) )
     *
     * Returns propagation angle in radians (0 = straight ahead).
     */
    static Real max_hoop_stress_angle(Real KI, Real KII) {
        if (std::abs(KII) < 1.0e-30) return 0.0;
        Real disc = std::sqrt(KI * KI + 8.0 * KII * KII);
        Real theta = 2.0 * std::atan2(KI - disc, 4.0 * KII);
        return theta;
    }

    /**
     * @brief Equivalent stress intensity factor (mixed mode).
     *
     * K_eq = KI * cos^3(theta/2) - 3 * KII * cos^2(theta/2) * sin(theta/2)
     * (evaluated at the max hoop stress angle)
     *
     * Simplified: K_eq = cos(theta/2) * [KI * cos^2(theta/2) - 1.5 * KII * sin(theta)]
     */
    static Real equivalent_sif(Real KI, Real KII) {
        Real theta = max_hoop_stress_angle(KI, KII);
        Real ht = theta * 0.5;
        Real ct = std::cos(ht);
        return ct * (KI * ct * ct - 1.5 * KII * std::sin(theta));
    }

    /**
     * @brief J-integral computation (plane strain).
     *
     * J = (1 - nu^2) / E * (KI^2 + KII^2) + (1 + nu) / E * KIII^2
     *
     * For 2D (KIII=0):
     *   J = (1 - nu^2) / E * (KI^2 + KII^2)
     */
    Real compute_J_integral(Real KI, Real KII, Real KIII = 0.0) const {
        Real factor_12 = (1.0 - nu_ * nu_) / E_;
        Real factor_3 = (1.0 + nu_) / E_;
        return factor_12 * (KI * KI + KII * KII) + factor_3 * KIII * KIII;
    }

    /**
     * @brief Determine if crack should propagate.
     * @return true if K_eq >= KIc
     */
    bool should_propagate(Real KI, Real KII) const {
        Real Keq = equivalent_sif(KI, KII);
        return Keq >= KIc_;
    }

    /**
     * @brief Update crack tip position given propagation.
     *
     * @param tip Current tip position {x, y}
     * @param crack_angle Current crack angle (radians)
     * @param KI, KII Stress intensity factors
     * @return New tip position and new crack angle as {x, y, new_angle}
     */
    std::array<Real, 3> propagate(Real tip_x, Real tip_y, Real crack_angle,
                                   Real KI, Real KII) const {
        Real dtheta = max_hoop_stress_angle(KI, KII);
        Real new_angle = crack_angle + dtheta;
        Real new_x = tip_x + da_ * std::cos(new_angle);
        Real new_y = tip_y + da_ * std::sin(new_angle);
        return {new_x, new_y, new_angle};
    }

    Real growth_increment() const { return da_; }
    Real fracture_toughness() const { return KIc_; }

private:
    Real E_ = 210.0e9;
    Real nu_ = 0.3;
    Real KIc_ = 50.0e6; // Pa*sqrt(m) — typical steel
    Real da_ = 0.001;    // 1 mm default growth increment
};

// ============================================================================
// 16c: Blast Loading — CONWEP Empirical Model
// ============================================================================

/**
 * @brief CONWEP blast loading model.
 *
 * Implements the Friedlander waveform for blast pressure-time history:
 *   P(t) = P_s * (1 - t/t_pos) * exp(-b * t / t_pos)   for 0 <= t <= t_pos
 *
 * Uses Hopkinson-Cranz scaling: Z = R / W^(1/3)
 * where R = standoff distance (m), W = charge mass TNT equivalent (kg).
 *
 * Empirical fits from Kingery-Bulmash (1984) for peak pressures and durations.
 */
class CONWEPBlast {
public:
    CONWEPBlast() = default;

    /**
     * @brief Set charge parameters.
     * @param charge_mass_kg Explosive charge mass in kg
     * @param tnt_equiv TNT equivalence factor (1.0 for TNT, 1.34 for C4, etc.)
     */
    void set_charge(Real charge_mass_kg, Real tnt_equiv = 1.0) {
        W_ = charge_mass_kg * tnt_equiv;
    }

    /**
     * @brief Set detonation point in 3D space.
     */
    void set_detonation_point(Real x, Real y, Real z) {
        det_point_ = {x, y, z};
    }

    /**
     * @brief Compute scaled distance Z = R / W^(1/3).
     */
    Real scaled_distance(Real standoff) const {
        Real W13 = std::cbrt(std::max(W_, 1.0e-30));
        return standoff / W13;
    }

    /**
     * @brief Empirical peak incident overpressure (Pa).
     *
     * Kingery-Bulmash fit (simplified):
     *   P_s = 0.084 / Z + 0.27 / Z^2 + 0.7 / Z^3  (bar)
     * converted to Pa (* 1e5).
     *
     * Valid for 0.05 < Z < 40 m/kg^(1/3).
     */
    Real peak_incident_pressure(Real standoff) const {
        Real Z = scaled_distance(standoff);
        if (Z < 0.05) Z = 0.05; // Clamp for near-field
        Real Ps_bar = 0.084 / Z + 0.27 / (Z * Z) + 0.7 / (Z * Z * Z);
        return Ps_bar * 1.0e5; // Convert to Pa
    }

    /**
     * @brief Peak reflected pressure.
     *
     * Reflection coefficient depends on angle of incidence alpha:
     *   Cr = 2 + 0.05 * (Ps/P0)  for normal incidence (alpha=0)
     *   Pr = Cr * Ps * cos^2(alpha) + Ps * (1 + cos^2(alpha) - 2*cos(alpha))
     *
     * Simplified oblique reflection (UFC 3-340-02):
     *   Pr = Ps * [2*cos(alpha) + (1 + 6*Ps/P0) * cos^2(alpha)] / (1 + Ps/P0)
     *   but clamped to range [Ps, Cr*Ps]
     */
    Real peak_reflected_pressure(Real standoff, Real angle_rad) const {
        Real Ps = peak_incident_pressure(standoff);
        Real ca = std::cos(angle_rad);
        if (ca < 0.0) ca = 0.0; // Behind the wave, no loading

        // Reflection coefficient for normal incidence
        Real Ps_over_P0 = Ps / P0_;
        Real Cr = 2.0 + 0.05 * Ps_over_P0;
        if (Cr > 8.0) Cr = 8.0; // Physical upper limit

        // Oblique reflection
        Real Pr = Ps * (2.0 * ca + (1.0 + 6.0 * Ps_over_P0) * ca * ca)
                  / (1.0 + Ps_over_P0);

        // Clamp between incident and max reflected
        if (Pr < Ps * ca) Pr = Ps * ca;
        if (Pr > Cr * Ps) Pr = Cr * Ps;

        return Pr;
    }

    /**
     * @brief Positive phase duration (seconds).
     *
     * Empirical fit: t_pos = 1.3e-3 * W^(1/3) * Z^(1/2) (seconds)
     */
    Real positive_phase_duration(Real standoff) const {
        Real Z = scaled_distance(standoff);
        Real W13 = std::cbrt(std::max(W_, 1.0e-30));
        return 1.3e-3 * W13 * std::sqrt(Z);
    }

    /**
     * @brief Friedlander decay coefficient.
     *
     * b is chosen so the impulse integral matches empirical data.
     * Approximation: b = 1.5 + 0.18 * Z
     */
    Real decay_coefficient(Real standoff) const {
        Real Z = scaled_distance(standoff);
        return 1.5 + 0.18 * Z;
    }

    /**
     * @brief Friedlander waveform: incident pressure vs time.
     *
     * P(t) = Ps * (1 - t/t_pos) * exp(-b * t / t_pos)   for 0 <= t <= t_pos
     * P(t) = 0  otherwise
     *
     * @param standoff Distance from charge to target (m)
     * @param time Time after blast arrival (s)
     * @return Overpressure (Pa)
     */
    Real friedlander_pressure(Real standoff, Real time) const {
        if (time < 0.0) return 0.0;
        Real t_pos = positive_phase_duration(standoff);
        if (time > t_pos) return 0.0;

        Real Ps = peak_incident_pressure(standoff);
        Real b = decay_coefficient(standoff);
        Real tau = time / t_pos;

        return Ps * (1.0 - tau) * std::exp(-b * tau);
    }

    /**
     * @brief Full reflected pressure-time history including angle of incidence.
     *
     * @param standoff Distance from charge to target (m)
     * @param angle_rad Angle of incidence (0 = normal, pi/2 = grazing)
     * @param time Time after blast arrival (s)
     * @return Reflected overpressure at surface (Pa)
     */
    Real compute_pressure(Real standoff, Real angle_rad, Real time) const {
        if (time < 0.0) return 0.0;
        Real t_pos = positive_phase_duration(standoff);
        if (time > t_pos) return 0.0;

        Real Pr = peak_reflected_pressure(standoff, angle_rad);
        Real b = decay_coefficient(standoff);
        Real tau = time / t_pos;

        return Pr * (1.0 - tau) * std::exp(-b * tau);
    }

    /**
     * @brief Arrival time of blast wave.
     *
     * t_a = R / c_blast, where c_blast ~ 340 * (1 + 0.7 * Ps/P0)^0.5
     * approximation for shock velocity.
     */
    Real arrival_time(Real standoff) const {
        Real Ps = peak_incident_pressure(standoff);
        Real c_shock = 340.0 * std::sqrt(1.0 + 0.7 * Ps / P0_);
        return standoff / c_shock;
    }

    /**
     * @brief Positive phase impulse (Pa*s).
     *
     * I = integral_0^t_pos P(t) dt = Ps * t_pos * [1/b - 1/b^2 * (1 - exp(-b))]
     */
    Real positive_impulse(Real standoff) const {
        Real Ps = peak_incident_pressure(standoff);
        Real t_pos = positive_phase_duration(standoff);
        Real b = decay_coefficient(standoff);
        if (std::abs(b) < 1.0e-10) return Ps * t_pos * 0.5;
        return Ps * t_pos * (1.0 / b - (1.0 - std::exp(-b)) / (b * b));
    }

    Real charge_mass_tnt() const { return W_; }

private:
    Real W_ = 1.0;                                 ///< TNT equivalent mass (kg)
    std::array<Real, 3> det_point_ = {0.0, 0.0, 0.0}; ///< Detonation point
    Real P0_ = 101325.0;                            ///< Ambient pressure (Pa)
};

// ============================================================================
// 16d: Airbag Simulation
// ============================================================================

/**
 * @brief Airbag gas inflation model.
 *
 * Uses ideal gas law with mass inflow, vent holes, and fabric porosity.
 *
 * Governing equations:
 *   dm/dt = m_dot_in - m_dot_vent - m_dot_porous
 *   P * V = m * R_gas * T / M_gas
 *   dT/dt = (1/(m*cv)) * [m_dot_in*cp*T_in - P*dV/dt - Q_loss
 *            - m_dot_vent*cp*T - m_dot_porous*cp*T]
 */
class AirbagModel {
public:
    AirbagModel() = default;

    /// Set gas properties
    void set_gas_properties(Real molar_mass, Real gamma) {
        M_gas_ = molar_mass;
        gamma_ = gamma;
        R_specific_ = R_universal_ / M_gas_;
        cv_ = R_specific_ / (gamma_ - 1.0);
        cp_ = gamma_ * cv_;
    }

    /// Set initial conditions
    void set_initial_conditions(Real volume, Real temperature, Real pressure) {
        V_ = volume;
        T_ = temperature;
        P_ = pressure;
        m_ = P_ * V_ / (R_specific_ * T_);
    }

    /// Set inflator mass flow rate curve (constant rate)
    void set_inflator(Real mass_flow_rate, Real gas_temperature) {
        m_dot_in_ = mass_flow_rate;
        T_in_ = gas_temperature;
    }

    /// Set vent hole parameters
    void set_vent(Real vent_area, Real discharge_coeff = 0.65) {
        A_vent_ = vent_area;
        Cd_vent_ = discharge_coeff;
    }

    /// Set fabric porosity parameters
    void set_porosity(Real permeability, Real fabric_area, Real fabric_thickness) {
        kappa_ = permeability;
        A_fabric_ = fabric_area;
        t_fabric_ = fabric_thickness;
    }

    /**
     * @brief Vent hole mass flow rate (orifice equation).
     *
     * For subsonic flow (P/P0 > critical ratio):
     *   m_dot = Cd * A * P * sqrt(2*gamma/((gamma-1)*R*T) * [(P0/P)^(2/g) - (P0/P)^((g+1)/g)])
     *
     * Simplified choked flow (if P > P_critical):
     *   m_dot = Cd * A * P / sqrt(R*T) * sqrt(gamma) * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
     */
    Real vent_mass_flow() const {
        if (A_vent_ <= 0.0 || P_ <= P_ambient_) return 0.0;

        Real P_ratio = P_ambient_ / P_;
        Real g = gamma_;
        Real P_crit = std::pow(2.0 / (g + 1.0), g / (g - 1.0));

        Real m_dot;
        if (P_ratio <= P_crit) {
            // Choked flow
            Real factor = std::sqrt(g) * std::pow(2.0 / (g + 1.0), (g + 1.0) / (2.0 * (g - 1.0)));
            m_dot = Cd_vent_ * A_vent_ * P_ * factor / std::sqrt(R_specific_ * T_);
        } else {
            // Subsonic flow
            Real term1 = std::pow(P_ratio, 2.0 / g);
            Real term2 = std::pow(P_ratio, (g + 1.0) / g);
            Real flow_func = std::sqrt(2.0 * g / (g - 1.0) * (term1 - term2));
            m_dot = Cd_vent_ * A_vent_ * P_ * flow_func / std::sqrt(R_specific_ * T_);
        }
        return m_dot;
    }

    /**
     * @brief Fabric porosity mass flow rate (Darcy's law).
     *
     * m_dot_porous = (kappa * A_fabric / (mu * t_fabric)) * (P - P_ambient) * rho
     * where mu is dynamic viscosity and rho = P / (R * T)
     *
     * Simplified: m_dot = C_perm * A_fabric * (P - P_ambient) / (R_specific * T)
     * where C_perm = kappa / (mu * t_fabric)
     */
    Real porous_mass_flow() const {
        if (kappa_ <= 0.0 || A_fabric_ <= 0.0 || P_ <= P_ambient_) return 0.0;

        Real dP = P_ - P_ambient_;
        Real mu = 1.8e-5; // Air dynamic viscosity at ~300K (Pa*s)
        Real C_perm = kappa_ / (mu * t_fabric_);
        Real rho = P_ / (R_specific_ * T_);
        return C_perm * A_fabric_ * dP * rho;
    }

    /**
     * @brief Advance the airbag state by one time step.
     *
     * Updates mass, temperature, and pressure using forward Euler.
     *
     * @param dt Time step (s)
     * @param new_volume Current airbag volume (m^3)
     */
    void step(Real dt, Real new_volume) {
        Real m_vent = vent_mass_flow();
        Real m_porous = porous_mass_flow();

        // Mass conservation
        Real dm = (m_dot_in_ - m_vent - m_porous) * dt;
        Real m_new = m_ + dm;
        if (m_new < 1.0e-30) m_new = 1.0e-30;

        // Volume change work
        Real dV = new_volume - V_;
        Real P_dV = P_ * dV;

        // Energy balance (first law of thermodynamics)
        // m*cv*dT = m_dot_in*cp*T_in*dt - P*dV - m_vent*cp*T*dt - m_porous*cp*T*dt
        Real energy_in = m_dot_in_ * cp_ * T_in_ * dt;
        Real energy_vent = m_vent * cp_ * T_ * dt;
        Real energy_porous = m_porous * cp_ * T_ * dt;
        Real dT = (energy_in - P_dV - energy_vent - energy_porous) / (m_new * cv_);

        // Update state
        m_ = m_new;
        T_ += dT;
        if (T_ < 200.0) T_ = 200.0; // Floor temperature
        V_ = new_volume;
        P_ = m_ * R_specific_ * T_ / V_;

        time_ += dt;
    }

    // Accessors
    Real pressure() const { return P_; }
    Real temperature() const { return T_; }
    Real mass() const { return m_; }
    Real volume() const { return V_; }
    Real time() const { return time_; }

    void set_ambient_pressure(Real P0) { P_ambient_ = P0; }

private:
    // Gas properties
    Real M_gas_ = 0.028;            ///< Molar mass (kg/mol) — N2 default
    Real gamma_ = 1.4;              ///< Heat capacity ratio
    Real R_specific_ = 296.8;       ///< Specific gas constant (J/kg-K)
    Real cv_ = 742.0;               ///< Cv (J/kg-K)
    Real cp_ = 1039.0;              ///< Cp (J/kg-K)

    // State variables
    Real m_ = 0.0;                  ///< Gas mass (kg)
    Real T_ = 300.0;                ///< Gas temperature (K)
    Real P_ = 101325.0;             ///< Gas pressure (Pa)
    Real V_ = 0.001;                ///< Bag volume (m^3)
    Real time_ = 0.0;               ///< Current time (s)

    // Inflator
    Real m_dot_in_ = 0.0;           ///< Inflator mass flow rate (kg/s)
    Real T_in_ = 1000.0;            ///< Inflator gas temperature (K)

    // Vent
    Real A_vent_ = 0.0;             ///< Vent hole area (m^2)
    Real Cd_vent_ = 0.65;           ///< Vent discharge coefficient

    // Porosity
    Real kappa_ = 0.0;              ///< Fabric permeability (m^2)
    Real A_fabric_ = 0.0;           ///< Total fabric area (m^2)
    Real t_fabric_ = 0.001;         ///< Fabric thickness (m)

    // Constants
    Real P_ambient_ = 101325.0;     ///< Ambient pressure (Pa)
    static constexpr Real R_universal_ = 8.314; ///< J/(mol*K)
};

// ============================================================================
// 16e: Seatbelt Dynamics
// ============================================================================

/**
 * @brief 1D seatbelt element with slip-ring DOFs.
 *
 * Models belt webbing as a 1D tension-only bar element with:
 * - Nonlinear stiffness from webbing stretch
 * - Rate-dependent effects
 * - Slip at belt anchor points (slip rings)
 */
class BeltElement {
public:
    BeltElement() = default;

    /**
     * @brief Initialize a belt segment.
     * @param initial_length Reference (free) length of belt segment (m)
     * @param stiffness Belt stiffness per unit length (N/m per m, i.e., N/m^2)
     * @param damping Belt damping coefficient (N*s/m)
     */
    void initialize(Real initial_length, Real stiffness, Real damping = 0.0) {
        L0_ = initial_length;
        L_ = initial_length;
        k_ = stiffness;
        c_ = damping;
        slip_total_ = 0.0;
        tension_ = 0.0;
    }

    /**
     * @brief Update belt state given current node positions.
     *
     * @param current_length Current length of belt segment
     * @param velocity_rate Rate of length change (dL/dt)
     * @param dt Time step
     */
    void update(Real current_length, Real velocity_rate, Real dt) {
        L_ = current_length;
        Real stretch = (L_ - L0_ + slip_total_) / L0_;

        // Tension only (no compression)
        if (stretch <= 0.0) {
            tension_ = 0.0;
            return;
        }

        // Rate-dependent tension
        Real static_tension = k_ * L0_ * stretch;
        Real dynamic_tension = c_ * velocity_rate;
        tension_ = static_tension + dynamic_tension;
        if (tension_ < 0.0) tension_ = 0.0;
    }

    /**
     * @brief Apply slip at a slip ring.
     *
     * Belt feeds through the ring, reducing effective length on one side
     * and increasing on the other.
     *
     * @param slip_amount Amount of belt feed-through (positive = pay-out)
     */
    void apply_slip(Real slip_amount) {
        slip_total_ += slip_amount;
    }

    /**
     * @brief Compute wrap-around friction at a contact point.
     *
     * Uses Euler's belt friction formula:
     *   T_tight / T_slack = exp(mu * theta)
     *
     * @param tension_slack Tension on slack side
     * @param friction_coeff Friction coefficient
     * @param wrap_angle Contact wrap angle (radians)
     * @return Tension on tight side
     */
    static Real belt_friction(Real tension_slack, Real friction_coeff, Real wrap_angle) {
        return tension_slack * std::exp(friction_coeff * wrap_angle);
    }

    Real tension() const { return tension_; }
    Real current_length() const { return L_; }
    Real initial_length() const { return L0_; }
    Real slip() const { return slip_total_; }
    Real stretch_ratio() const { return (L0_ > 0.0) ? L_ / L0_ : 1.0; }

private:
    Real L0_ = 1.0;            ///< Initial length (m)
    Real L_ = 1.0;             ///< Current length (m)
    Real k_ = 50000.0;         ///< Stiffness per unit length (N/m^2)
    Real c_ = 100.0;           ///< Damping coefficient (N*s/m)
    Real tension_ = 0.0;       ///< Current tension (N)
    Real slip_total_ = 0.0;    ///< Accumulated slip (m)
};

/**
 * @brief Seatbelt retractor model.
 *
 * Features:
 * - Spool-out: belt pays out from reel at low tension
 * - Locking: engages at specified deceleration or belt pull rate
 * - Pretensioner: pyrotechnic retraction (applies force to tighten belt)
 * - Load limiter: caps tension after threshold to limit chest loading
 */
class Retractor {
public:
    Retractor() = default;

    /**
     * @brief Set spool-out parameters.
     * @param max_pullout Maximum belt that can be spooled out (m)
     * @param spool_stiffness Rotational stiffness of spool (N*m/rad)
     * @param spool_radius Spool radius (m)
     */
    void set_spool(Real max_pullout, Real spool_stiffness, Real spool_radius) {
        max_pullout_ = max_pullout;
        spool_k_ = spool_stiffness;
        spool_r_ = spool_radius;
    }

    /**
     * @brief Set locking parameters.
     * @param lock_decel Deceleration threshold for locking (m/s^2)
     * @param lock_pullrate Belt pull rate threshold for locking (m/s)
     */
    void set_locking(Real lock_decel, Real lock_pullrate) {
        lock_decel_ = lock_decel;
        lock_pullrate_ = lock_pullrate;
    }

    /**
     * @brief Set pretensioner parameters.
     * @param fire_time Time at which pretensioner fires (s)
     * @param force Pretensioner force (N)
     * @param stroke Maximum retraction stroke (m)
     */
    void set_pretensioner(Real fire_time, Real force, Real stroke) {
        pt_fire_time_ = fire_time;
        pt_force_ = force;
        pt_stroke_ = stroke;
        has_pretensioner_ = true;
    }

    /**
     * @brief Set load limiter parameters.
     * @param force_limit Maximum belt force (N)
     */
    void set_load_limiter(Real force_limit) {
        force_limit_ = force_limit;
        has_load_limiter_ = true;
    }

    /**
     * @brief Update retractor state.
     *
     * @param time Current simulation time (s)
     * @param belt_tension Current belt tension (N)
     * @param deceleration Current vehicle deceleration (m/s^2)
     * @param pull_rate Belt pull rate (m/s)
     * @param dt Time step (s)
     * @return Effective belt force after retractor effects (N)
     */
    Real update(Real time, Real belt_tension, Real deceleration,
                Real pull_rate, Real dt) {
        // Check locking condition
        if (!locked_) {
            if (deceleration >= lock_decel_ || pull_rate >= lock_pullrate_) {
                locked_ = true;
            }
        }

        // Pretensioner activation
        Real pt_force = 0.0;
        if (has_pretensioner_ && time >= pt_fire_time_ && pt_retracted_ < pt_stroke_) {
            pt_force = pt_force_;
            Real retract = pt_force / std::max(spool_k_, 1.0) * dt;
            pt_retracted_ += retract;
            if (pt_retracted_ > pt_stroke_) pt_retracted_ = pt_stroke_;
            locked_ = true; // Pretensioner always locks
        }

        Real effective_force = belt_tension;

        if (!locked_) {
            // Spool-out: low resistance, belt pays out
            if (pullout_ < max_pullout_ && pull_rate > 0.0) {
                Real delta_pull = pull_rate * dt;
                pullout_ += delta_pull;
                if (pullout_ > max_pullout_) pullout_ = max_pullout_;
            }
            // Spool resistance force
            Real spool_force = spool_k_ * (pullout_ / spool_r_) * spool_r_;
            effective_force = std::min(belt_tension, spool_force);
        } else {
            // Locked: add pretensioner force
            effective_force = belt_tension + pt_force;
        }

        // Load limiter
        if (has_load_limiter_ && effective_force > force_limit_) {
            Real excess = effective_force - force_limit_;
            limiter_payout_ += (excess / force_limit_) * dt;
            effective_force = force_limit_;
        }

        return effective_force;
    }

    bool is_locked() const { return locked_; }
    Real pullout() const { return pullout_; }
    Real pretensioner_retracted() const { return pt_retracted_; }
    Real limiter_payout() const { return limiter_payout_; }

private:
    // Spool
    Real max_pullout_ = 0.3;       ///< Max spool-out (m)
    Real spool_k_ = 500.0;         ///< Spool stiffness (N*m/rad)
    Real spool_r_ = 0.02;          ///< Spool radius (m)
    Real pullout_ = 0.0;           ///< Current pullout amount (m)

    // Locking
    Real lock_decel_ = 5.0;        ///< Lock deceleration threshold (m/s^2)
    Real lock_pullrate_ = 0.3;     ///< Lock pull rate threshold (m/s)
    bool locked_ = false;

    // Pretensioner
    bool has_pretensioner_ = false;
    Real pt_fire_time_ = 0.01;     ///< Fire time (s)
    Real pt_force_ = 2000.0;       ///< Pretensioner force (N)
    Real pt_stroke_ = 0.06;        ///< Max retraction (m)
    Real pt_retracted_ = 0.0;      ///< Amount retracted so far (m)

    // Load limiter
    bool has_load_limiter_ = false;
    Real force_limit_ = 4000.0;    ///< Force limit (N)
    Real limiter_payout_ = 0.0;    ///< Belt paid out by limiter (m)
};

// ============================================================================
// 16f: Advanced ALE — Eulerian Solver, Cut-Cell, Turbulence
// ============================================================================

/**
 * @brief Pure Eulerian solver on a fixed Cartesian grid.
 *
 * Material flows through a fixed mesh. State variables (density, momentum,
 * energy) are advected using upwind scheme.
 *
 * Conservation form:
 *   d(rho)/dt + div(rho * u) = 0
 *   d(rho*u)/dt + div(rho*u*u + P*I) = 0
 *   d(rho*E)/dt + div((rho*E + P)*u) = 0
 */
class EulerianSolver {
public:
    EulerianSolver() = default;

    /**
     * @brief Initialize a 1D Eulerian grid.
     * @param num_cells Number of cells
     * @param dx Cell size (m)
     */
    void initialize(int num_cells, Real dx) {
        ncells_ = num_cells;
        dx_ = dx;
        rho_.assign(ncells_, 1.225);      // Air density
        vel_.assign(ncells_, 0.0);         // Velocity
        pressure_.assign(ncells_, 101325.0); // Pressure
        energy_.assign(ncells_, 0.0);       // Internal energy
        gamma_ = 1.4;

        // Initialize internal energy from pressure
        for (int i = 0; i < ncells_; ++i) {
            energy_[i] = pressure_[i] / ((gamma_ - 1.0) * rho_[i]);
        }
    }

    /// Set state for a cell
    void set_cell_state(int cell, Real rho, Real vel, Real pressure) {
        rho_[cell] = rho;
        vel_[cell] = vel;
        pressure_[cell] = pressure;
        energy_[cell] = pressure / ((gamma_ - 1.0) * rho);
    }

    /**
     * @brief First-order upwind advection step.
     *
     * Uses donor-cell (upwind) scheme for stability:
     *   F_{i+1/2} = u_{i+1/2} > 0 ? q_i * u : q_{i+1} * u
     *
     * @param dt Time step (s)
     */
    void step_upwind(Real dt) {
        std::vector<Real> rho_new = rho_;
        std::vector<Real> mom_new(ncells_);
        std::vector<Real> ene_new(ncells_);

        // Convert to conservative variables
        std::vector<Real> mom(ncells_), tot_e(ncells_);
        for (int i = 0; i < ncells_; ++i) {
            mom[i] = rho_[i] * vel_[i];
            tot_e[i] = rho_[i] * (energy_[i] + 0.5 * vel_[i] * vel_[i]);
        }

        Real dtdx = dt / dx_;

        for (int i = 1; i < ncells_ - 1; ++i) {
            // Interface velocities
            Real u_right = 0.5 * (vel_[i] + vel_[i + 1]);
            Real u_left = 0.5 * (vel_[i - 1] + vel_[i]);

            // Density flux (upwind)
            Real flux_rho_r = (u_right > 0) ? rho_[i] * u_right : rho_[i + 1] * u_right;
            Real flux_rho_l = (u_left > 0) ? rho_[i - 1] * u_left : rho_[i] * u_left;
            rho_new[i] = rho_[i] - dtdx * (flux_rho_r - flux_rho_l);

            // Momentum flux (upwind + pressure)
            Real flux_mom_r = (u_right > 0) ? mom[i] * u_right : mom[i + 1] * u_right;
            flux_mom_r += pressure_[i + 1]; // Pressure at right face (simplified)
            Real flux_mom_l = (u_left > 0) ? mom[i - 1] * u_left : mom[i] * u_left;
            flux_mom_l += pressure_[i]; // Pressure at left face
            // Corrected pressure term
            flux_mom_r = (u_right > 0 ? mom[i] * u_right : mom[i + 1] * u_right)
                         + 0.5 * (pressure_[i] + pressure_[i + 1]);
            flux_mom_l = (u_left > 0 ? mom[i - 1] * u_left : mom[i] * u_left)
                         + 0.5 * (pressure_[i - 1] + pressure_[i]);
            mom_new[i] = mom[i] - dtdx * (flux_mom_r - flux_mom_l);

            // Energy flux (upwind)
            Real flux_e_r = (u_right > 0)
                ? (tot_e[i] + pressure_[i]) * u_right
                : (tot_e[i + 1] + pressure_[i + 1]) * u_right;
            Real flux_e_l = (u_left > 0)
                ? (tot_e[i - 1] + pressure_[i - 1]) * u_left
                : (tot_e[i] + pressure_[i]) * u_left;
            ene_new[i] = tot_e[i] - dtdx * (flux_e_r - flux_e_l);
        }

        // Update primitive variables
        for (int i = 1; i < ncells_ - 1; ++i) {
            rho_[i] = std::max(rho_new[i], 1.0e-10);
            vel_[i] = mom_new[i] / rho_[i];
            Real KE = 0.5 * rho_[i] * vel_[i] * vel_[i];
            energy_[i] = std::max((ene_new[i] - KE) / rho_[i], 1.0e-10);
            pressure_[i] = (gamma_ - 1.0) * rho_[i] * energy_[i];
        }
    }

    // Accessors
    Real density(int cell) const { return rho_[cell]; }
    Real velocity(int cell) const { return vel_[cell]; }
    Real pressure(int cell) const { return pressure_[cell]; }
    Real internal_energy(int cell) const { return energy_[cell]; }
    int num_cells() const { return ncells_; }

    /**
     * @brief CFL-based stable time step.
     * dt = CFL * dx / max(|u| + c)
     */
    Real compute_stable_dt(Real cfl = 0.5) const {
        Real max_speed = 0.0;
        for (int i = 0; i < ncells_; ++i) {
            Real c = std::sqrt(std::max(gamma_ * pressure_[i] / rho_[i], 0.0));
            Real speed = std::abs(vel_[i]) + c;
            if (speed > max_speed) max_speed = speed;
        }
        if (max_speed < 1.0e-30) return 1.0e-3;
        return cfl * dx_ / max_speed;
    }

private:
    int ncells_ = 0;
    Real dx_ = 1.0;
    Real gamma_ = 1.4;
    std::vector<Real> rho_;
    std::vector<Real> vel_;
    std::vector<Real> pressure_;
    std::vector<Real> energy_;
};

/**
 * @brief Multi-material cell with volume fraction tracking.
 *
 * Tracks volume fractions of multiple materials within a single cell.
 * Uses piecewise-linear interface reconstruction (PLIC) for sharp interfaces.
 */
class CutCellMethod {
public:
    static constexpr int MAX_MATERIALS = 4;

    CutCellMethod() = default;

    /**
     * @brief Initialize a multi-material cell.
     * @param num_materials Number of materials (max 4)
     */
    void initialize(int num_materials) {
        nmat_ = std::min(num_materials, MAX_MATERIALS);
        for (int i = 0; i < MAX_MATERIALS; ++i) {
            vf_[i] = (i < nmat_) ? 1.0 / nmat_ : 0.0;
            rho_[i] = 1000.0;
        }
    }

    /// Set volume fraction for material i
    void set_volume_fraction(int mat, Real vf) {
        if (mat >= 0 && mat < nmat_) vf_[mat] = vf;
    }

    /// Set density for material i
    void set_density(int mat, Real rho) {
        if (mat >= 0 && mat < nmat_) rho_[mat] = rho;
    }

    Real volume_fraction(int mat) const { return (mat >= 0 && mat < nmat_) ? vf_[mat] : 0.0; }
    Real density(int mat) const { return (mat >= 0 && mat < nmat_) ? rho_[mat] : 0.0; }

    /// Mixture density: rho_mix = sum(vf_i * rho_i)
    Real mixture_density() const {
        Real sum = 0.0;
        for (int i = 0; i < nmat_; ++i) sum += vf_[i] * rho_[i];
        return sum;
    }

    /**
     * @brief PLIC interface reconstruction.
     *
     * Given volume fraction and interface normal, compute the line constant
     * d such that the plane n.x = d cuts the unit cell to give the correct
     * volume fraction.
     *
     * For 2D unit cell [0,1]^2 with normal (nx, ny):
     *   - Sort |nx|, |ny| to get the "effective" shape
     *   - Use analytical formula for cut area vs. line position
     *
     * @param vf Volume fraction of material below the interface
     * @param nx, ny Interface normal components
     * @return Line constant d (offset)
     */
    static Real plic_reconstruct(Real vf, Real nx, Real ny) {
        // Normalize normal
        Real mag = std::sqrt(nx * nx + ny * ny);
        if (mag < 1.0e-30) return 0.5;
        nx /= mag;
        ny /= mag;

        // Use absolute values (symmetry)
        Real anx = std::abs(nx);
        Real any = std::abs(ny);

        // Make anx <= any for simpler formulas
        if (anx > any) std::swap(anx, any);

        // For a unit square, area below line anx*x + any*y = d
        // Various cases: see Scardovelli & Zaleski (2000)
        // Simplified: use bisection for robustness
        Real d_lo = 0.0, d_hi = anx + any;
        for (int iter = 0; iter < 50; ++iter) {
            Real d = 0.5 * (d_lo + d_hi);
            Real area = plic_area(d, anx, any);
            if (area < vf) d_lo = d;
            else d_hi = d;
        }
        return 0.5 * (d_lo + d_hi);
    }

    /**
     * @brief Stabilize small cut cells by merging with neighbors.
     *
     * If a material's volume fraction is below a threshold, its mass
     * is redistributed to neighbors.
     *
     * @param threshold Minimum volume fraction (default 0.01)
     */
    void stabilize_small_cells(Real threshold = 0.01) {
        // Redistribute mass from very small volume fractions
        Real total_mass = 0.0;
        for (int i = 0; i < nmat_; ++i) total_mass += vf_[i] * rho_[i];

        for (int i = 0; i < nmat_; ++i) {
            if (vf_[i] > 0.0 && vf_[i] < threshold) {
                Real mass_to_redistribute = vf_[i] * rho_[i];
                vf_[i] = 0.0;

                // Find largest volume fraction material to receive mass
                int max_mat = -1;
                Real max_vf = 0.0;
                for (int j = 0; j < nmat_; ++j) {
                    if (j != i && vf_[j] > max_vf) {
                        max_vf = vf_[j];
                        max_mat = j;
                    }
                }
                if (max_mat >= 0) {
                    vf_[max_mat] += mass_to_redistribute / rho_[max_mat];
                }
            }
        }

        // Renormalize volume fractions to sum to 1
        Real sum = 0.0;
        for (int i = 0; i < nmat_; ++i) sum += vf_[i];
        if (sum > 0.0) {
            for (int i = 0; i < nmat_; ++i) vf_[i] /= sum;
        }
    }

    int num_materials() const { return nmat_; }

private:
    int nmat_ = 2;
    Real vf_[MAX_MATERIALS] = {0.5, 0.5, 0.0, 0.0};
    Real rho_[MAX_MATERIALS] = {1000.0, 1000.0, 1000.0, 1000.0};

    /// Compute area of unit square below line anx*x + any*y = d
    /// where 0 <= anx <= any
    static Real plic_area(Real d, Real anx, Real any) {
        if (d <= 0.0) return 0.0;
        if (d >= anx + any) return 1.0;

        if (d < anx) {
            // Triangle in corner
            return d * d / (2.0 * anx * any);
        } else if (d < any) {
            // Trapezoid
            return (d - 0.5 * anx) / any;
        } else {
            // 1 - triangle in opposite corner
            Real rem = anx + any - d;
            return 1.0 - rem * rem / (2.0 * anx * any);
        }
    }
};

/**
 * @brief k-epsilon turbulence model.
 *
 * Standard model constants (Launder & Sharma):
 *   C_mu = 0.09, C_1 = 1.44, C_2 = 1.92, sigma_k = 1.0, sigma_eps = 1.3
 *
 * Transport equations:
 *   dk/dt = P_k - epsilon + div((nu + nu_t/sigma_k) * grad(k))
 *   deps/dt = C_1 * eps/k * P_k - C_2 * eps^2/k + div((nu + nu_t/sigma_eps) * grad(eps))
 *
 * where nu_t = C_mu * k^2 / epsilon (turbulent kinematic viscosity)
 */
class TurbulenceModel {
public:
    TurbulenceModel() = default;

    /**
     * @brief Initialize k-epsilon model at a point.
     * @param k0 Initial turbulent kinetic energy (m^2/s^2)
     * @param eps0 Initial dissipation rate (m^2/s^3)
     * @param rho Fluid density (kg/m^3)
     * @param mu Molecular dynamic viscosity (Pa*s)
     */
    void initialize(Real k0, Real eps0, Real rho, Real mu) {
        k_ = k0;
        epsilon_ = eps0;
        rho_ = rho;
        mu_mol_ = mu;
    }

    /**
     * @brief Compute turbulent viscosity.
     * mu_t = rho * C_mu * k^2 / epsilon
     */
    Real turbulent_viscosity() const {
        if (epsilon_ < 1.0e-30) return 0.0;
        return rho_ * C_mu_ * k_ * k_ / epsilon_;
    }

    /// Effective viscosity: mu_eff = mu + mu_t
    Real effective_viscosity() const {
        return mu_mol_ + turbulent_viscosity();
    }

    /**
     * @brief Advance k-epsilon equations by one time step.
     *
     * @param production P_k = mu_t * S^2 where S is strain rate magnitude
     * @param dt Time step
     */
    void step(Real production, Real dt) {
        // k equation: dk/dt = P_k - epsilon
        Real dk = (production - rho_ * epsilon_) * dt / rho_;
        k_ += dk;
        if (k_ < 1.0e-10) k_ = 1.0e-10;

        // epsilon equation: deps/dt = C1*eps/k*P_k - C2*eps^2/k
        Real eps_over_k = epsilon_ / std::max(k_, 1.0e-30);
        Real deps = (C1_ * eps_over_k * production / rho_ - C2_ * epsilon_ * eps_over_k) * dt;
        epsilon_ += deps;
        if (epsilon_ < 1.0e-10) epsilon_ = 1.0e-10;
    }

    /**
     * @brief Standard wall function: compute wall shear stress.
     *
     * In the log-law region (y+ > 11.63):
     *   u+ = (1/kappa) * ln(E * y+)
     *   where kappa = 0.41 (von Karman), E = 9.793
     *
     * Friction velocity: u_tau = C_mu^{1/4} * k^{1/2}
     * Wall shear: tau_w = rho * u_tau * u / u+
     *
     * @param u_parallel Velocity parallel to wall (m/s)
     * @param y_distance Distance from wall (m)
     * @return Wall shear stress (Pa)
     */
    Real wall_shear_stress(Real u_parallel, Real y_distance) const {
        Real u_tau = std::pow(C_mu_, 0.25) * std::sqrt(std::max(k_, 0.0));
        if (u_tau < 1.0e-30) return 0.0;

        Real y_plus = rho_ * u_tau * y_distance / mu_mol_;

        Real u_plus;
        if (y_plus <= 11.63) {
            // Viscous sublayer
            u_plus = y_plus;
        } else {
            // Log-law region
            u_plus = (1.0 / kappa_vk_) * std::log(E_wall_ * y_plus);
        }

        if (u_plus < 1.0e-30) return 0.0;
        return rho_ * u_tau * std::abs(u_parallel) / u_plus;
    }

    // Accessors
    Real k() const { return k_; }
    Real epsilon() const { return epsilon_; }
    Real turbulent_kinetic_energy() const { return k_; }
    Real dissipation_rate() const { return epsilon_; }

    /// Turbulent length scale: l_t = C_mu^{3/4} * k^{3/2} / epsilon
    Real turbulent_length_scale() const {
        if (epsilon_ < 1.0e-30) return 0.0;
        return std::pow(C_mu_, 0.75) * std::pow(k_, 1.5) / epsilon_;
    }

    /// Turbulent time scale: t_t = k / epsilon
    Real turbulent_time_scale() const {
        if (epsilon_ < 1.0e-30) return 1.0e30;
        return k_ / epsilon_;
    }

private:
    Real k_ = 1.0;                 ///< Turbulent kinetic energy (m^2/s^2)
    Real epsilon_ = 1.0;           ///< Dissipation rate (m^2/s^3)
    Real rho_ = 1.225;             ///< Density (kg/m^3)
    Real mu_mol_ = 1.8e-5;         ///< Molecular viscosity (Pa*s)

    // Standard k-epsilon constants
    static constexpr Real C_mu_ = 0.09;
    static constexpr Real C1_ = 1.44;
    static constexpr Real C2_ = 1.92;
    static constexpr Real sigma_k_ = 1.0;
    static constexpr Real sigma_eps_ = 1.3;

    // Wall function constants
    static constexpr Real kappa_vk_ = 0.41;   ///< von Karman constant
    static constexpr Real E_wall_ = 9.793;     ///< Wall function log-law constant
};

// ============================================================================
// 16g: Adaptive Mesh Refinement
// ============================================================================

/**
 * @brief Mesh quality metric for a quadrilateral/hexahedral element.
 */
struct MeshQuality {
    Real aspect_ratio = 1.0;    ///< Max edge / min edge
    Real jacobian_ratio = 1.0;  ///< Min Jacobian / max Jacobian
    Real skewness = 0.0;        ///< Deviation from orthogonality [0, 1]
    Real warpage = 0.0;         ///< Out-of-plane warping [0, 1]

    /// Overall quality score [0 = worst, 1 = best]
    Real quality_score() const {
        Real ar_score = (aspect_ratio > 0.0) ? 1.0 / aspect_ratio : 0.0;
        Real jac_score = std::max(jacobian_ratio, 0.0);
        Real skew_score = 1.0 - skewness;
        return (ar_score + jac_score + skew_score) / 3.0;
    }
};

/**
 * @brief Adaptive Mesh Refinement manager.
 *
 * Implements:
 * - Zienkiewicz-Zhu (ZZ) superconvergent patch recovery error estimator
 * - h-refinement (element subdivision)
 * - Coarsening (element merging)
 * - Hanging node constraint management
 */
class AMRManager {
public:
    AMRManager() = default;

    /**
     * @brief Initialize AMR with mesh data.
     * @param num_elements Number of elements
     * @param num_nodes Number of nodes
     */
    void initialize(int num_elements, int num_nodes) {
        nelem_ = num_elements;
        nnodes_ = num_nodes;
        error_.assign(nelem_, 0.0);
        refinement_level_.assign(nelem_, 0);
        marked_refine_.assign(nelem_, false);
        marked_coarsen_.assign(nelem_, false);
        element_active_.assign(nelem_, true);
    }

    /**
     * @brief Zienkiewicz-Zhu stress recovery error estimator.
     *
     * For each element, the error is estimated as:
     *   e_i^2 = integral |sigma* - sigma_h|^2 dOmega
     *
     * where sigma* is the recovered (smoothed) stress and sigma_h is the
     * FE stress. The recovered stress is obtained by least-squares fitting
     * a polynomial to the stress at superconvergent points.
     *
     * Simplified implementation: compare element stress to average of
     * neighbor stresses (nodal averaging).
     *
     * @param element_stress Stress at each element centroid (von Mises or component)
     * @param element_volume Element volumes for weighting
     */
    void compute_zz_error(const std::vector<Real>& element_stress,
                          const std::vector<Real>& element_volume) {
        if (element_stress.size() != static_cast<size_t>(nelem_)) return;

        // Compute global average stress (as proxy for recovered stress)
        Real total_volume = 0.0;
        Real weighted_stress = 0.0;
        for (int i = 0; i < nelem_; ++i) {
            if (!element_active_[i]) continue;
            Real vol = (i < static_cast<int>(element_volume.size())) ? element_volume[i] : 1.0;
            weighted_stress += element_stress[i] * vol;
            total_volume += vol;
        }
        Real avg_stress = (total_volume > 0.0) ? weighted_stress / total_volume : 0.0;

        // Energy norm of global error
        Real global_error_sq = 0.0;
        for (int i = 0; i < nelem_; ++i) {
            if (!element_active_[i]) { error_[i] = 0.0; continue; }
            Real vol = (i < static_cast<int>(element_volume.size())) ? element_volume[i] : 1.0;
            Real diff = element_stress[i] - avg_stress;
            error_[i] = std::sqrt(diff * diff * vol);
            global_error_sq += diff * diff * vol;
        }

        global_error_norm_ = std::sqrt(global_error_sq);
    }

    /**
     * @brief Compute ZZ error using neighbor averaging for recovered stress.
     *
     * @param element_stress Stress at element centroids
     * @param element_volume Element volumes
     * @param neighbors Neighbor list for each element (adjacency)
     */
    void compute_zz_error_with_neighbors(
        const std::vector<Real>& element_stress,
        const std::vector<Real>& element_volume,
        const std::vector<std::vector<int>>& neighbors)
    {
        if (element_stress.size() != static_cast<size_t>(nelem_)) return;

        Real global_error_sq = 0.0;

        for (int i = 0; i < nelem_; ++i) {
            if (!element_active_[i]) { error_[i] = 0.0; continue; }

            // Recovered stress = average of this element + neighbors
            Real sum_stress = element_stress[i];
            Real sum_vol = (i < static_cast<int>(element_volume.size())) ? element_volume[i] : 1.0;
            int count = 1;

            if (i < static_cast<int>(neighbors.size())) {
                for (int j : neighbors[i]) {
                    if (j >= 0 && j < nelem_ && element_active_[j]) {
                        Real vj = (j < static_cast<int>(element_volume.size())) ? element_volume[j] : 1.0;
                        sum_stress += element_stress[j] * vj;
                        sum_vol += vj;
                        count++;
                    }
                }
            }

            Real recovered = sum_stress / std::max(sum_vol, 1.0e-30);
            Real vol_i = (i < static_cast<int>(element_volume.size())) ? element_volume[i] : 1.0;
            Real diff = element_stress[i] - recovered;
            error_[i] = std::sqrt(std::abs(diff * diff * vol_i));
            global_error_sq += diff * diff * vol_i;
        }

        global_error_norm_ = std::sqrt(global_error_sq);
    }

    /**
     * @brief Mark elements for refinement and coarsening.
     *
     * Uses threshold-based marking:
     * - Refine if error > refine_fraction * max_error
     * - Coarsen if error < coarsen_fraction * max_error
     *
     * @param refine_frac Fraction of max error above which to refine (e.g. 0.5)
     * @param coarsen_frac Fraction of max error below which to coarsen (e.g. 0.1)
     * @param max_level Maximum refinement level allowed
     */
    void mark_elements(Real refine_frac = 0.5, Real coarsen_frac = 0.1, int max_level = 4) {
        Real max_error = *std::max_element(error_.begin(), error_.end());
        if (max_error < 1.0e-30) return;

        Real refine_threshold = refine_frac * max_error;
        Real coarsen_threshold = coarsen_frac * max_error;

        for (int i = 0; i < nelem_; ++i) {
            marked_refine_[i] = false;
            marked_coarsen_[i] = false;
            if (!element_active_[i]) continue;

            if (error_[i] > refine_threshold && refinement_level_[i] < max_level) {
                marked_refine_[i] = true;
            } else if (error_[i] < coarsen_threshold && refinement_level_[i] > 0) {
                marked_coarsen_[i] = true;
            }
        }
    }

    /**
     * @brief Perform h-refinement by subdividing marked elements.
     *
     * Each refined quadrilateral is split into 4 children.
     * Returns the number of new elements created.
     *
     * Note: This is a logical refinement that updates the element list.
     * Actual node creation and connectivity update is mesh-specific.
     */
    int refine() {
        int new_elements = 0;
        int old_nelem = nelem_;

        for (int i = 0; i < old_nelem; ++i) {
            if (!marked_refine_[i] || !element_active_[i]) continue;

            // Deactivate parent
            element_active_[i] = false;

            // Create 4 children (quad subdivision)
            int child_level = refinement_level_[i] + 1;
            for (int c = 0; c < 4; ++c) {
                error_.push_back(error_[i] * 0.25); // Error distributed
                refinement_level_.push_back(child_level);
                marked_refine_.push_back(false);
                marked_coarsen_.push_back(false);
                element_active_.push_back(true);
                parent_.push_back(i);
                new_elements++;
            }
        }

        nelem_ += new_elements;
        return new_elements;
    }

    /**
     * @brief Coarsen marked elements (merge children back to parent).
     * Returns number of elements removed.
     */
    int coarsen() {
        int removed = 0;
        // Simple implementation: deactivate elements marked for coarsening
        // In a real implementation, we'd merge sibling children back to parent
        for (int i = 0; i < nelem_; ++i) {
            if (marked_coarsen_[i] && element_active_[i] && refinement_level_[i] > 0) {
                element_active_[i] = false;
                removed++;
            }
        }
        return removed;
    }

    /**
     * @brief Compute hanging node constraint coefficients.
     *
     * A hanging node on an edge midpoint is constrained:
     *   u_hanging = 0.5 * (u_left + u_right)
     *
     * @param hanging_node The hanging node index
     * @param left_node, right_node The two parent edge nodes
     * @return Constraint coefficients {0.5, 0.5}
     */
    static std::array<Real, 2> hanging_node_constraint(int /*hanging_node*/,
                                                        int /*left_node*/,
                                                        int /*right_node*/) {
        return {0.5, 0.5};
    }

    /**
     * @brief Compute mesh quality for a quad element given 4 corner coordinates.
     */
    static MeshQuality compute_quality(const std::array<std::array<Real, 2>, 4>& corners) {
        MeshQuality q;

        // Edge lengths
        Real edges[4];
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            Real dx = corners[j][0] - corners[i][0];
            Real dy = corners[j][1] - corners[i][1];
            edges[i] = std::sqrt(dx * dx + dy * dy);
        }

        Real min_edge = *std::min_element(edges, edges + 4);
        Real max_edge = *std::max_element(edges, edges + 4);
        q.aspect_ratio = (min_edge > 1.0e-30) ? max_edge / min_edge : 1.0e10;

        // Jacobian at each corner (cross product of edge vectors)
        Real jacs[4];
        for (int i = 0; i < 4; ++i) {
            int prev = (i + 3) % 4;
            int next = (i + 1) % 4;
            Real dx1 = corners[next][0] - corners[i][0];
            Real dy1 = corners[next][1] - corners[i][1];
            Real dx2 = corners[prev][0] - corners[i][0];
            Real dy2 = corners[prev][1] - corners[i][1];
            jacs[i] = dx1 * dy2 - dy1 * dx2;
        }

        Real min_jac = *std::min_element(jacs, jacs + 4);
        Real max_jac = *std::max_element(jacs, jacs + 4);
        q.jacobian_ratio = (std::abs(max_jac) > 1.0e-30) ? min_jac / max_jac : 0.0;

        // Skewness: deviation of angles from 90 degrees
        Real max_angle_dev = 0.0;
        for (int i = 0; i < 4; ++i) {
            int prev = (i + 3) % 4;
            int next = (i + 1) % 4;
            Real dx1 = corners[next][0] - corners[i][0];
            Real dy1 = corners[next][1] - corners[i][1];
            Real dx2 = corners[prev][0] - corners[i][0];
            Real dy2 = corners[prev][1] - corners[i][1];
            Real len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
            Real len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (len1 > 1.0e-30 && len2 > 1.0e-30) {
                Real cosang = (dx1 * dx2 + dy1 * dy2) / (len1 * len2);
                cosang = std::max(-1.0, std::min(1.0, cosang));
                Real angle = std::acos(cosang);
                Real dev = std::abs(angle - M_PI / 2.0) / (M_PI / 2.0);
                if (dev > max_angle_dev) max_angle_dev = dev;
            }
        }
        q.skewness = max_angle_dev;

        return q;
    }

    // Accessors
    int num_elements() const { return nelem_; }
    int num_active_elements() const {
        int count = 0;
        for (int i = 0; i < nelem_; ++i) if (element_active_[i]) count++;
        return count;
    }
    Real element_error(int i) const { return error_[i]; }
    int refinement_level(int i) const { return refinement_level_[i]; }
    bool is_active(int i) const { return element_active_[i]; }
    bool is_marked_refine(int i) const { return marked_refine_[i]; }
    bool is_marked_coarsen(int i) const { return marked_coarsen_[i]; }
    Real global_error_norm() const { return global_error_norm_; }

private:
    int nelem_ = 0;
    int nnodes_ = 0;
    std::vector<Real> error_;
    std::vector<int> refinement_level_;
    std::vector<bool> marked_refine_;
    std::vector<bool> marked_coarsen_;
    std::vector<bool> element_active_;
    std::vector<int> parent_;
    Real global_error_norm_ = 0.0;
};

// ============================================================================
// 16h: Draping Analysis
// ============================================================================

/**
 * @brief Draping analysis for composite fiber mapping onto curved mold surfaces.
 *
 * Implements kinematic (pin-point) draping based on the fishnet algorithm:
 * 1. Start from a pin point on the mold surface
 * 2. Generate geodesic paths along warp and weft directions
 * 3. Compute fiber angles (shear deformation) at each cell
 * 4. Check for shear locking (manufacturing limit)
 *
 * Reference: Mack & Taylor (1956), van der Ween (1991)
 */
class DrapingAnalysis {
public:
    DrapingAnalysis() = default;

    /**
     * @brief Initialize with a flat mold surface (for testing).
     *
     * Sets up a rectangular grid of nx * ny elements on a surface
     * that may have curvature specified by a height function z(x,y).
     */
    void initialize_flat(int nx, int ny, Real lx, Real ly) {
        nx_ = nx;
        ny_ = ny;
        nelem_ = nx * ny;
        Real dx = lx / nx;
        Real dy = ly / ny;

        // Generate nodes
        int nnx = nx + 1, nny = ny + 1;
        nodes_.resize(nnx * nny);
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int idx = j * nnx + i;
                nodes_[idx] = {i * dx, j * dy, 0.0};
            }
        }

        // Initialize fiber angles to zero (undraped)
        fiber_angle_.assign(nelem_, 0.0);
        shear_angle_.assign(nelem_, 0.0);
        draped_.assign(nelem_, false);
    }

    /**
     * @brief Initialize with curved surface nodes.
     * @param nodes 3D coordinates of surface nodes
     * @param nx, ny Grid dimensions
     */
    void initialize_surface(const std::vector<std::array<Real, 3>>& nodes, int nx, int ny) {
        nx_ = nx;
        ny_ = ny;
        nelem_ = nx * ny;
        nodes_ = nodes;
        fiber_angle_.assign(nelem_, 0.0);
        shear_angle_.assign(nelem_, 0.0);
        draped_.assign(nelem_, false);
    }

    /**
     * @brief Set the pin point (draping origin).
     * @param elem_i, elem_j Element indices of pin point
     * @param initial_angle Initial fiber angle at pin (radians)
     */
    void set_pin_point(int elem_i, int elem_j, Real initial_angle = 0.0) {
        pin_i_ = elem_i;
        pin_j_ = elem_j;
        initial_angle_ = initial_angle;
    }

    /// Set maximum shear angle before locking (radians)
    void set_lock_angle(Real lock_angle) { lock_angle_ = lock_angle; }

    /**
     * @brief Perform kinematic draping from pin point.
     *
     * Uses the fishnet algorithm:
     * 1. At pin point, set fiber angle = initial_angle
     * 2. Propagate along warp (i-direction) and weft (j-direction)
     * 3. At each cell, compute geodesic distance to determine fiber path
     * 4. Compute shear angle from deviation of fiber from right angle
     *
     * The shear angle gamma at each cell is:
     *   gamma = pi/2 - alpha
     * where alpha is the angle between warp and weft fibers.
     *
     * For a curved surface, the shear depends on the Gaussian curvature.
     */
    void drape() {
        if (nelem_ == 0) return;

        int nnx = nx_ + 1;

        // Start from pin point
        int pin_elem = pin_j_ * nx_ + pin_i_;
        if (pin_elem < 0 || pin_elem >= nelem_) {
            pin_i_ = nx_ / 2;
            pin_j_ = ny_ / 2;
            pin_elem = pin_j_ * nx_ + pin_i_;
        }

        fiber_angle_[pin_elem] = initial_angle_;
        shear_angle_[pin_elem] = 0.0;
        draped_[pin_elem] = true;

        // Propagate outward from pin point using BFS-like approach
        // Process in layers: first the pin row/column, then expanding
        for (int di = -std::max(pin_i_, nx_ - pin_i_); di <= std::max(pin_i_, nx_ - pin_i_); ++di) {
            for (int dj = -std::max(pin_j_, ny_ - pin_j_); dj <= std::max(pin_j_, ny_ - pin_j_); ++dj) {
                int ei = pin_i_ + di;
                int ej = pin_j_ + dj;
                if (ei < 0 || ei >= nx_ || ej < 0 || ej >= ny_) continue;
                int elem = ej * nx_ + ei;
                if (draped_[elem]) continue;

                // Compute fiber angle based on surface curvature
                // Get element corner nodes
                int n0 = ej * nnx + ei;
                int n1 = n0 + 1;
                int n2 = n0 + nnx + 1;
                int n3 = n0 + nnx;

                if (n2 >= static_cast<int>(nodes_.size())) continue;

                // Element edge vectors
                Real e1x = nodes_[n1][0] - nodes_[n0][0];
                Real e1y = nodes_[n1][1] - nodes_[n0][1];
                Real e1z = nodes_[n1][2] - nodes_[n0][2];

                Real e2x = nodes_[n3][0] - nodes_[n0][0];
                Real e2y = nodes_[n3][1] - nodes_[n0][1];
                Real e2z = nodes_[n3][2] - nodes_[n0][2];

                // Angle between edges (should be ~90 for unsheared)
                Real len1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
                Real len2 = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);

                Real cos_alpha = 0.0;
                if (len1 > 1.0e-30 && len2 > 1.0e-30) {
                    cos_alpha = (e1x * e2x + e1y * e2y + e1z * e2z) / (len1 * len2);
                    cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));
                }

                Real alpha = std::acos(cos_alpha);
                Real shear = M_PI / 2.0 - alpha; // Deviation from right angle

                // Fiber angle: initial angle + accumulated rotation from surface curvature
                Real curvature_rotation = compute_surface_rotation(ei, ej, pin_i_, pin_j_);

                fiber_angle_[elem] = initial_angle_ + curvature_rotation;
                shear_angle_[elem] = std::abs(shear);
                draped_[elem] = true;
            }
        }
    }

    /**
     * @brief Compute geodesic distance between two points on the surface.
     *
     * Uses Dijkstra-like propagation along mesh edges.
     * Simplified: sum of edge lengths along shortest path on grid.
     */
    Real geodesic_distance(int elem1_i, int elem1_j, int elem2_i, int elem2_j) const {
        int nnx = nx_ + 1;
        // Get centroids
        auto centroid = [&](int ei, int ej) -> std::array<Real, 3> {
            int n0 = ej * nnx + ei;
            int n1 = n0 + 1;
            int n2 = n0 + nnx + 1;
            int n3 = n0 + nnx;
            return {
                0.25 * (nodes_[n0][0] + nodes_[n1][0] + nodes_[n2][0] + nodes_[n3][0]),
                0.25 * (nodes_[n0][1] + nodes_[n1][1] + nodes_[n2][1] + nodes_[n3][1]),
                0.25 * (nodes_[n0][2] + nodes_[n1][2] + nodes_[n2][2] + nodes_[n3][2])
            };
        };

        // Simple Manhattan-path geodesic on the surface mesh
        Real total_dist = 0.0;
        int ci = elem1_i, cj = elem1_j;
        while (ci != elem2_i || cj != elem2_j) {
            auto c1 = centroid(ci, cj);
            int ni = ci, nj = cj;
            if (ci < elem2_i) ni++;
            else if (ci > elem2_i) ni--;
            else if (cj < elem2_j) nj++;
            else if (cj > elem2_j) nj--;

            auto c2 = centroid(ni, nj);
            Real dx = c2[0] - c1[0], dy = c2[1] - c1[1], dz = c2[2] - c1[2];
            total_dist += std::sqrt(dx * dx + dy * dy + dz * dz);
            ci = ni;
            cj = nj;
        }
        return total_dist;
    }

    /**
     * @brief Check if any element exceeds the shear lock angle.
     * @return Number of locked elements
     */
    int check_locking() const {
        int count = 0;
        for (int i = 0; i < nelem_; ++i) {
            if (draped_[i] && shear_angle_[i] > lock_angle_) count++;
        }
        return count;
    }

    // Accessors
    Real fiber_angle(int elem) const { return fiber_angle_[elem]; }
    Real shear_angle(int elem) const { return shear_angle_[elem]; }
    bool is_draped(int elem) const { return draped_[elem]; }
    int num_elements() const { return nelem_; }
    int num_draped() const {
        int c = 0;
        for (int i = 0; i < nelem_; ++i) if (draped_[i]) c++;
        return c;
    }

    /// Get all fiber angles for output
    std::vector<Real> get_fiber_angles() const { return fiber_angle_; }
    std::vector<Real> get_shear_angles() const { return shear_angle_; }

private:
    int nx_ = 0, ny_ = 0;
    int nelem_ = 0;
    int pin_i_ = 0, pin_j_ = 0;
    Real initial_angle_ = 0.0;
    Real lock_angle_ = 1.0472; // 60 degrees default

    std::vector<std::array<Real, 3>> nodes_;
    std::vector<Real> fiber_angle_;
    std::vector<Real> shear_angle_;
    std::vector<bool> draped_;

    /**
     * @brief Compute accumulated surface rotation from pin to target element.
     *
     * Integrates the geodesic curvature along the path. For a flat surface
     * this is zero; for curved surfaces it depends on Gaussian curvature.
     */
    Real compute_surface_rotation(int ei, int ej, int pi, int pj) const {
        // Estimate surface curvature from height variations
        int nnx = nx_ + 1;
        Real total_rotation = 0.0;

        // Walk from pin to target, accumulating rotation
        int ci = pi, cj = pj;
        while (ci != ei || cj != ej) {
            int n0 = cj * nnx + ci;
            if (n0 + nnx + 1 >= static_cast<int>(nodes_.size())) break;

            // Local curvature from height field second derivative
            Real z00 = nodes_[n0][2];
            Real z10 = nodes_[n0 + 1][2];
            Real z01 = nodes_[n0 + nnx][2];
            Real z11 = nodes_[n0 + nnx + 1][2];

            // Approximate Gaussian curvature contribution
            Real twist = (z11 - z10 - z01 + z00);
            Real dx = nodes_[n0 + 1][0] - nodes_[n0][0];
            Real dy = nodes_[n0 + nnx][1] - nodes_[n0][1];
            if (std::abs(dx) > 1.0e-30 && std::abs(dy) > 1.0e-30) {
                total_rotation += twist / (dx * dy);
            }

            // Step toward target
            if (ci < ei) ci++;
            else if (ci > ei) ci--;
            else if (cj < ej) cj++;
            else if (cj > ej) cj--;
        }

        return total_rotation;
    }
};

} // namespace advanced
} // namespace nxs
