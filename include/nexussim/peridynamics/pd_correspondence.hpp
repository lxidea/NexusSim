#pragma once

/**
 * @file pd_correspondence.hpp
 * @brief Non-ordinary state-based PD using the correspondence model
 *
 * Computes deformation gradient F from peridynamic states, then uses
 * classical constitutive models (linear elastic, neo-Hookean, SVK)
 * to obtain Cauchy stress. Includes zero-energy mode stabilization.
 *
 * Reference: Silling (2007), Breitenfeld et al. (2014)
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>

namespace nxs {
namespace pd {

// ============================================================================
// Mat3 — lightweight 3x3 matrix for correspondence model
// ============================================================================

struct Mat3 {
    Real data[3][3] = {};

    KOKKOS_INLINE_FUNCTION
    Real& operator()(int i, int j) { return data[i][j]; }

    KOKKOS_INLINE_FUNCTION
    Real operator()(int i, int j) const { return data[i][j]; }

    KOKKOS_INLINE_FUNCTION
    Mat3 operator*(const Mat3& B) const {
        Mat3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < 3; ++k)
                    sum += data[i][k] * B.data[k][j];
                C.data[i][j] = sum;
            }
        return C;
    }

    KOKKOS_INLINE_FUNCTION
    Mat3 operator+(const Mat3& B) const {
        Mat3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                C.data[i][j] = data[i][j] + B.data[i][j];
        return C;
    }

    KOKKOS_INLINE_FUNCTION
    Mat3 operator-(const Mat3& B) const {
        Mat3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                C.data[i][j] = data[i][j] - B.data[i][j];
        return C;
    }

    KOKKOS_INLINE_FUNCTION
    Mat3 operator*(Real s) const {
        Mat3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                C.data[i][j] = data[i][j] * s;
        return C;
    }

    KOKKOS_INLINE_FUNCTION
    Mat3 transpose() const {
        Mat3 T;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T.data[i][j] = data[j][i];
        return T;
    }

    KOKKOS_INLINE_FUNCTION
    Real determinant() const {
        return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
             - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
             + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
    }

    KOKKOS_INLINE_FUNCTION
    Mat3 inverse() const {
        Real det = determinant();
        Real inv_det = (Kokkos::fabs(det) > 1e-200) ? 1.0 / det : 0.0;
        Mat3 inv;
        inv.data[0][0] = (data[1][1] * data[2][2] - data[1][2] * data[2][1]) * inv_det;
        inv.data[0][1] = (data[0][2] * data[2][1] - data[0][1] * data[2][2]) * inv_det;
        inv.data[0][2] = (data[0][1] * data[1][2] - data[0][2] * data[1][1]) * inv_det;
        inv.data[1][0] = (data[1][2] * data[2][0] - data[1][0] * data[2][2]) * inv_det;
        inv.data[1][1] = (data[0][0] * data[2][2] - data[0][2] * data[2][0]) * inv_det;
        inv.data[1][2] = (data[0][2] * data[1][0] - data[0][0] * data[1][2]) * inv_det;
        inv.data[2][0] = (data[1][0] * data[2][1] - data[1][1] * data[2][0]) * inv_det;
        inv.data[2][1] = (data[0][1] * data[2][0] - data[0][0] * data[2][1]) * inv_det;
        inv.data[2][2] = (data[0][0] * data[1][1] - data[0][1] * data[1][0]) * inv_det;
        return inv;
    }

    KOKKOS_INLINE_FUNCTION
    Real trace() const {
        return data[0][0] + data[1][1] + data[2][2];
    }
};

KOKKOS_INLINE_FUNCTION
Mat3 mat3_identity() {
    Mat3 I;
    I.data[0][0] = 1.0; I.data[1][1] = 1.0; I.data[2][2] = 1.0;
    return I;
}

KOKKOS_INLINE_FUNCTION
Mat3 mat3_zero() {
    return Mat3{};
}

KOKKOS_INLINE_FUNCTION
Mat3 mat3_sym(const Mat3& A) {
    Mat3 S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S.data[i][j] = 0.5 * (A.data[i][j] + A.data[j][i]);
    return S;
}

KOKKOS_INLINE_FUNCTION
Mat3 mat3_dev(const Mat3& A) {
    Mat3 D = A;
    Real tr = A.trace() / 3.0;
    D.data[0][0] -= tr;
    D.data[1][1] -= tr;
    D.data[2][2] -= tr;
    return D;
}

// ============================================================================
// Constitutive model type for correspondence
// ============================================================================

enum class CorrespondenceModel {
    LinearElastic,
    NeoHookean,
    SaintVenantKirchhoff
};

// ============================================================================
// Constitutive stress functions
// ============================================================================

KOKKOS_INLINE_FUNCTION
Mat3 stress_linear_elastic(const Mat3& F, Real E, Real nu) {
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));

    // Small strain: epsilon = 0.5*(F + F^T) - I
    Mat3 eps = mat3_sym(F) - mat3_identity();

    Real tr_eps = eps.trace();
    Mat3 I = mat3_identity();

    // sigma = lambda * tr(eps) * I + 2 * mu * eps
    Mat3 sigma;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sigma(i, j) = lambda * tr_eps * I(i, j) + 2.0 * mu * eps(i, j);
    return sigma;
}

KOKKOS_INLINE_FUNCTION
Mat3 stress_neo_hookean(const Mat3& F, Real E, Real nu) {
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));

    Real J = F.determinant();
    if (J < 1e-10) J = 1e-10;

    // B = F * F^T (left Cauchy-Green)
    Mat3 Ft = F.transpose();
    Mat3 B = F * Ft;

    Mat3 I = mat3_identity();
    Real lnJ = Kokkos::log(J);

    // sigma = (mu/J)(B - I) + (lambda * lnJ / J) * I
    Mat3 sigma;
    Real inv_J = 1.0 / J;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sigma(i, j) = mu * inv_J * (B(i, j) - I(i, j))
                         + lambda * lnJ * inv_J * I(i, j);
    return sigma;
}

KOKKOS_INLINE_FUNCTION
Mat3 stress_svk(const Mat3& F, Real E, Real nu) {
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));

    Real J = F.determinant();
    if (Kokkos::fabs(J) < 1e-10) J = 1e-10;

    // Green-Lagrange strain: E_GL = 0.5*(F^T F - I)
    Mat3 Ft = F.transpose();
    Mat3 C = Ft * F;
    Mat3 I = mat3_identity();
    Mat3 E_GL;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            E_GL(i, j) = 0.5 * (C(i, j) - I(i, j));

    // 2nd Piola-Kirchhoff: S = lambda * tr(E) * I + 2 * mu * E
    Real tr_E = E_GL.trace();
    Mat3 S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S(i, j) = lambda * tr_E * I(i, j) + 2.0 * mu * E_GL(i, j);

    // Cauchy stress: sigma = (1/J) F S F^T
    Mat3 FS = F * S;
    Mat3 sigma_J = FS * Ft;
    Real inv_J = 1.0 / J;
    return sigma_J * inv_J;
}

// ============================================================================
// PDCorrespondenceForce
// ============================================================================

class PDCorrespondenceForce {
public:
    PDCorrespondenceForce() = default;

    void initialize(const std::vector<PDMaterial>& materials,
                    CorrespondenceModel model = CorrespondenceModel::LinearElastic,
                    Real stabilization_param = 0.1) {
        num_materials_ = materials.size();
        model_ = model;
        stabilization_param_ = stabilization_param;

        E_ = PDScalarView("corr_E", num_materials_);
        nu_ = PDScalarView("corr_nu", num_materials_);

        auto E_host = Kokkos::create_mirror_view(E_);
        auto nu_host = Kokkos::create_mirror_view(nu_);

        for (Index i = 0; i < num_materials_; ++i) {
            E_host(i) = materials[i].E;
            nu_host(i) = materials[i].nu;
        }

        Kokkos::deep_copy(E_, E_host);
        Kokkos::deep_copy(nu_, nu_host);
    }

    /**
     * @brief Compute shape tensor K for all particles
     *        K_i = sum_j w(|xi|) xi (x) xi V_j
     */
    void compute_shape_tensor(PDParticleSystem& particles, PDNeighborList& neighbors) {
        Index num_particles = particles.num_particles();

        if (shape_tensor_.extent(0) != static_cast<size_t>(num_particles)) {
            shape_tensor_ = Kokkos::View<Real*[9]>("shape_tensor", num_particles);
            shape_inv_ = Kokkos::View<Real*[9]>("shape_inv", num_particles);
        }

        auto x0 = particles.x0();
        auto volume = particles.volume();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();

        auto K = shape_tensor_;
        auto K_inv = shape_inv_;

        Kokkos::parallel_for("compute_shape_tensor", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    for (int c = 0; c < 9; ++c) K(i, c) = 0.0;
                    for (int c = 0; c < 9; ++c) K_inv(i, c) = 0.0;
                    return;
                }

                Mat3 K_mat = mat3_zero();
                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real Vj = volume(j);

                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };

                    // K += w * xi (x) xi * Vj
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            K_mat(a, b) += w * xi[a] * xi[b] * Vj;
                }

                // Store K in flat [9] layout
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        K(i, a * 3 + b) = K_mat(a, b);

                // Compute and store K^{-1}
                Mat3 K_inv_mat = K_mat.inverse();
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        K_inv(i, a * 3 + b) = K_inv_mat(a, b);
            });
    }

    /**
     * @brief Compute deformation gradient F for all particles
     *        F_i = (sum_j w Y (x) xi V_j) K^{-1}
     */
    void compute_deformation_gradient(PDParticleSystem& particles, PDNeighborList& neighbors) {
        Index num_particles = particles.num_particles();

        if (def_grad_.extent(0) != static_cast<size_t>(num_particles)) {
            def_grad_ = Kokkos::View<Real*[9]>("def_grad", num_particles);
        }

        auto x = particles.x();
        auto x0 = particles.x0();
        auto volume = particles.volume();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();

        auto K_inv = shape_inv_;
        auto F_view = def_grad_;

        Kokkos::parallel_for("compute_deformation_gradient", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    for (int c = 0; c < 9; ++c) F_view(i, c) = 0.0;
                    return;
                }

                // Reconstruct K_inv
                Mat3 K_inv_mat;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        K_inv_mat(a, b) = K_inv(i, a * 3 + b);

                Mat3 N = mat3_zero(); // N = sum w * Y (x) xi * Vj
                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real Vj = volume(j);

                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };

                    // Y = x_j - x_i (deformed relative position)
                    Real Y[3] = {
                        x(j, 0) - x(i, 0),
                        x(j, 1) - x(i, 1),
                        x(j, 2) - x(i, 2)
                    };

                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            N(a, b) += w * Y[a] * xi[b] * Vj;
                }

                // F = N * K^{-1}
                Mat3 F_mat = N * K_inv_mat;

                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        F_view(i, a * 3 + b) = F_mat(a, b);
            });
    }

    /**
     * @brief Compute forces using the correspondence model
     *
     * For each particle i, compute Cauchy stress from F, then
     * force state T[xi] = w * sigma * K^{-1} * xi
     */
    void compute_forces(PDParticleSystem& particles, PDNeighborList& neighbors,
                        const std::vector<PDMaterial>& materials) {
        compute_shape_tensor(particles, neighbors);
        compute_deformation_gradient(particles, neighbors);

        particles.zero_forces();

        auto x = particles.x();
        auto x0 = particles.x0();
        auto f = particles.f();
        auto volume = particles.volume();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();

        auto F_view = def_grad_;
        auto K_inv = shape_inv_;
        auto E_v = E_;
        auto nu_v = nu_;
        auto model = model_;

        Index num_particles = particles.num_particles();

        Kokkos::parallel_for("compute_corr_forces", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Index mat_i = material_id(i);
                Real E_i = E_v(mat_i);
                Real nu_i = nu_v(mat_i);

                // Reconstruct F
                Mat3 F_mat;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        F_mat(a, b) = F_view(i, a * 3 + b);

                // Compute Cauchy stress based on model
                Mat3 sigma;
                switch (model) {
                    case CorrespondenceModel::NeoHookean:
                        sigma = stress_neo_hookean(F_mat, E_i, nu_i);
                        break;
                    case CorrespondenceModel::SaintVenantKirchhoff:
                        sigma = stress_svk(F_mat, E_i, nu_i);
                        break;
                    default:
                        sigma = stress_linear_elastic(F_mat, E_i, nu_i);
                        break;
                }

                // Reconstruct K_inv
                Mat3 K_inv_mat;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        K_inv_mat(a, b) = K_inv(i, a * 3 + b);

                // P = sigma * K_inv (force-state prefactor)
                Mat3 P = sigma * K_inv_mat;

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                Real fi[3] = {0.0, 0.0, 0.0};

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real Vj = volume(j);

                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };

                    // T[xi] = w * P * xi
                    Real T[3] = {0.0, 0.0, 0.0};
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            T[a] += w * P(a, b) * xi[b];

                    fi[0] += T[0] * Vj;
                    fi[1] += T[1] * Vj;
                    fi[2] += T[2] * Vj;
                }

                Kokkos::atomic_add(&f(i, 0), fi[0]);
                Kokkos::atomic_add(&f(i, 1), fi[1]);
                Kokkos::atomic_add(&f(i, 2), fi[2]);
            });
    }

    /**
     * @brief Apply zero-energy mode stabilization
     *
     * Adds a penalty force to suppress hourglass (zero-energy) modes.
     * Penalizes the non-uniform part of the deformation:
     *   f_stab = G_s * (Y - F*xi) / |xi|^2 * w * Vj
     */
    void apply_stabilization(PDParticleSystem& particles, PDNeighborList& neighbors) {
        auto x = particles.x();
        auto x0 = particles.x0();
        auto f = particles.f();
        auto volume = particles.volume();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto F_view = def_grad_;
        auto E_v = E_;
        auto nu_v = nu_;
        Real G_s_factor = stabilization_param_;

        Index num_particles = particles.num_particles();

        Kokkos::parallel_for("apply_stabilization", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Index mat_i = material_id(i);
                Real E_i = E_v(mat_i);
                Real nu_i = nu_v(mat_i);
                Real G_i = E_i / (2.0 * (1.0 + nu_i));
                Real G_s = G_s_factor * G_i;

                // Reconstruct F
                Mat3 F_mat;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        F_mat(a, b) = F_view(i, a * 3 + b);

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                Real fi[3] = {0.0, 0.0, 0.0};

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real Vj = volume(j);
                    Real xi_len = bond_length(bond_idx);

                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };

                    // Y = x_j - x_i
                    Real Y[3] = {
                        x(j, 0) - x(i, 0),
                        x(j, 1) - x(i, 1),
                        x(j, 2) - x(i, 2)
                    };

                    // F*xi
                    Real Fxi[3] = {0.0, 0.0, 0.0};
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            Fxi[a] += F_mat(a, b) * xi[b];

                    // Non-uniform part: z = Y - F*xi
                    Real z[3] = {
                        Y[0] - Fxi[0],
                        Y[1] - Fxi[1],
                        Y[2] - Fxi[2]
                    };

                    // Stabilization force: f_s = G_s * z / |xi|^2 * w * Vj
                    Real inv_xi2 = 1.0 / (xi_len * xi_len + 1e-30);
                    fi[0] += G_s * z[0] * inv_xi2 * w * Vj;
                    fi[1] += G_s * z[1] * inv_xi2 * w * Vj;
                    fi[2] += G_s * z[2] * inv_xi2 * w * Vj;
                }

                Kokkos::atomic_add(&f(i, 0), fi[0]);
                Kokkos::atomic_add(&f(i, 1), fi[1]);
                Kokkos::atomic_add(&f(i, 2), fi[2]);
            });
    }

    // Accessors
    Kokkos::View<Real*[9]>& shape_tensor() { return shape_tensor_; }
    Kokkos::View<Real*[9]>& shape_inverse() { return shape_inv_; }
    Kokkos::View<Real*[9]>& deformation_gradient() { return def_grad_; }
    CorrespondenceModel model() const { return model_; }

private:
    Index num_materials_ = 0;
    CorrespondenceModel model_ = CorrespondenceModel::LinearElastic;
    Real stabilization_param_ = 0.1;

    PDScalarView E_;
    PDScalarView nu_;

    Kokkos::View<Real*[9]> shape_tensor_;  // K per particle [N][9]
    Kokkos::View<Real*[9]> shape_inv_;     // K^{-1} per particle
    Kokkos::View<Real*[9]> def_grad_;      // F per particle
};

} // namespace pd
} // namespace nxs
