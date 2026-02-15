#pragma once

/**
 * @file composite_progressive_failure.hpp
 * @brief Progressive ply failure, FPF analysis, and strength envelope computation
 *
 * Features:
 * - Progressive damage with stiffness degradation and ABD recomputation
 * - First-ply-failure (FPF) load multiplier for different criteria
 * - Strength envelope computation (Nxx-Nyy and Nxx-Nxy planes)
 *
 * Degradation models:
 * - Ply Discount: Failed ply Q_bar set to zero
 * - Selective Discount: Only failed mode directions degraded
 * - Gradual: Smooth degradation via damage variable
 *
 * Reference: Reddy, "Mechanics of Laminated Composite Plates and Shells", Ch 12
 */

#include <nexussim/physics/composite_layup.hpp>
#include <nexussim/physics/composite_utils.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace physics {

enum class DegradationModel {
    Ply_Discount,       ///< Zero all Q_bar for failed ply
    Selective_Discount, ///< Zero only failed-mode stiffness
    Gradual             ///< Smooth degradation via factor
};

enum class FailureCriterion {
    Hashin,
    TsaiWu
};

struct PlyFailureStatus {
    bool failed;
    int failure_mode;        ///< 0=none, 1=fiber_tension, 2=fiber_compression, 3=matrix_tension, 4=matrix_compression
    Real failure_index;      ///< Max failure index
    Real degradation_factor; ///< 0=fully failed, 1=intact

    PlyFailureStatus()
        : failed(false), failure_mode(0), failure_index(0.0), degradation_factor(1.0) {}
};

struct ProgressiveFailureResult {
    PlyFailureStatus ply_status[CompositeLaminate::MAX_PLIES];
    Real degraded_A[9];
    Real degraded_B[9];
    Real degraded_D[9];
    Real load_multiplier;
    bool laminate_failed;
    int num_failed_plies;

    ProgressiveFailureResult() : load_multiplier(1.0), laminate_failed(false), num_failed_plies(0) {
        for (int i = 0; i < 9; ++i) { degraded_A[i] = 0.0; degraded_B[i] = 0.0; degraded_D[i] = 0.0; }
    }
};

struct FPFResult {
    Real load_multiplier;
    int critical_ply;
    int failure_mode;
    FailureCriterion criterion;

    FPFResult() : load_multiplier(0.0), critical_ply(-1), failure_mode(0), criterion(FailureCriterion::Hashin) {}
};

struct StrengthPoint {
    Real Nxx, Nyy, Nxy;
    int critical_ply;
    int failure_mode;

    StrengthPoint() : Nxx(0.0), Nyy(0.0), Nxy(0.0), critical_ply(-1), failure_mode(0) {}
};

class CompositeProgressiveFailure {
public:
    static constexpr int MAX_PLIES = CompositeLaminate::MAX_PLIES;

    CompositeProgressiveFailure()
        : Xt_(1500e6), Xc_(1200e6), Yt_(50e6), Yc_(200e6)
        , S12_(70e6), S23_(40e6)
        , degradation_model_(DegradationModel::Ply_Discount)
        , residual_stiffness_(0.0)
        , F12_star_(-0.5) {}

    void set_failure_params(Real Xt, Real Xc, Real Yt, Real Yc, Real S12, Real S23) {
        Xt_ = Xt; Xc_ = Xc; Yt_ = Yt; Yc_ = Yc; S12_ = S12; S23_ = S23;
    }

    void set_degradation_model(DegradationModel model) { degradation_model_ = model; }
    void set_residual_stiffness(Real factor) { residual_stiffness_ = factor; }
    void set_F12_star(Real val) { F12_star_ = val; }

    /**
     * @brief Single-step evaluation: compute ply failure indices for given deformation
     */
    ProgressiveFailureResult evaluate(const CompositeLaminate& lam,
                                       const Real* eps0,
                                       const Real* kappa) const {
        ProgressiveFailureResult result;

        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        for (int k = 0; k < lam.num_plies(); ++k) {
            Real z_mid = (z_bottom[k] + z_top[k]) / 2.0;

            Real Qbar[9];
            composite_detail::compute_Qbar(lam.ply(k), Qbar);

            // Global strain at ply midpoint
            Real strain_g[3];
            for (int i = 0; i < 3; ++i)
                strain_g[i] = eps0[i] + z_mid * kappa[i];

            // Global stress
            Real stress_g[3];
            for (int i = 0; i < 3; ++i) {
                stress_g[i] = 0.0;
                for (int j = 0; j < 3; ++j)
                    stress_g[i] += Qbar[i * 3 + j] * strain_g[j];
            }

            // Transform to local (material) coords
            Real theta = lam.ply(k).angle * constants::pi<Real> / 180.0;
            Real stress_l[3];
            composite_detail::transform_to_local(theta, stress_g, stress_l);

            // Evaluate Hashin failure indices
            Real s11 = stress_l[0], s22 = stress_l[1], t12 = stress_l[2];
            Real max_fi = 0.0;
            int mode = 0;

            // Mode 1: Fiber tension
            if (s11 > 0.0) {
                Real f1 = (s11 / Xt_) * (s11 / Xt_) + (t12 / S12_) * (t12 / S12_);
                if (f1 > max_fi) { max_fi = f1; mode = 1; }
            }
            // Mode 2: Fiber compression
            if (s11 < 0.0) {
                Real f2 = (s11 / Xc_) * (s11 / Xc_);
                if (f2 > max_fi) { max_fi = f2; mode = 2; }
            }
            // Mode 3: Matrix tension
            if (s22 > 0.0) {
                Real f3 = (s22 / Yt_) * (s22 / Yt_) + (t12 / S12_) * (t12 / S12_);
                if (f3 > max_fi) { max_fi = f3; mode = 3; }
            }
            // Mode 4: Matrix compression
            if (s22 < 0.0) {
                Real f4 = (s22 / (2.0 * S23_)) * (s22 / (2.0 * S23_))
                         + ((Yc_ / (2.0 * S23_)) * (Yc_ / (2.0 * S23_)) - 1.0) * (s22 / Yc_)
                         + (t12 / S12_) * (t12 / S12_);
                if (f4 > max_fi) { max_fi = f4; mode = 4; }
            }

            result.ply_status[k].failure_index = max_fi;
            result.ply_status[k].failure_mode = mode;
            if (max_fi >= 1.0) {
                result.ply_status[k].failed = true;
                result.ply_status[k].degradation_factor = residual_stiffness_;
                result.num_failed_plies++;
            }
        }

        result.laminate_failed = (result.num_failed_plies == lam.num_plies());

        // Compute degraded ABD
        compute_degraded_abd(lam, result);

        return result;
    }

    /**
     * @brief Progressive analysis: incrementally increase load until all plies fail
     *
     * @return Number of failure events (load levels where new plies fail)
     */
    int progressive_analysis(const CompositeLaminate& lam,
                              const Real* N_app, const Real* M_app,
                              ProgressiveFailureResult* results,
                              int max_results = 10) const {
        // Start with intact laminate
        bool ply_failed[MAX_PLIES] = {};
        int failure_modes[MAX_PLIES] = {};
        int num_events = 0;

        // Working copy of ply degradation factors
        Real deg_factor[MAX_PLIES];
        for (int k = 0; k < lam.num_plies(); ++k) deg_factor[k] = 1.0;

        Real cumulative_lambda = 0.0;

        while (num_events < max_results) {
            // Build current (degraded) ABD
            Real cur_A[9] = {}, cur_B[9] = {}, cur_D[9] = {};
            Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
            composite_detail::compute_z_coords(lam, z_bottom, z_top);

            for (int k = 0; k < lam.num_plies(); ++k) {
                if (ply_failed[k] && degradation_model_ == DegradationModel::Ply_Discount)
                    continue;

                Real Qbar[9];
                compute_degraded_Qbar(lam.ply(k), ply_failed[k], failure_modes[k],
                                       deg_factor[k], Qbar);

                Real zb = z_bottom[k], zt = z_top[k];
                Real dz = zt - zb;
                Real dz2 = zt * zt - zb * zb;
                Real dz3 = zt * zt * zt - zb * zb * zb;

                for (int i = 0; i < 9; ++i) {
                    cur_A[i] += Qbar[i] * dz;
                    cur_B[i] += 0.5 * Qbar[i] * dz2;
                    cur_D[i] += (1.0 / 3.0) * Qbar[i] * dz3;
                }
            }

            // Build and invert current ABD
            Real abd[36];
            for (int i = 0; i < 36; ++i) abd[i] = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    abd[i * 6 + j] = cur_A[i * 3 + j];
                    abd[i * 6 + (j + 3)] = cur_B[i * 3 + j];
                    abd[(i + 3) * 6 + j] = cur_B[i * 3 + j];
                    abd[(i + 3) * 6 + (j + 3)] = cur_D[i * 3 + j];
                }
            }

            Real abd_inv[36];
            if (!composite_detail::invert_6x6(abd, abd_inv)) break;

            // Compute unit-load deformation
            Real rhs[6] = {N_app[0], N_app[1], N_app[2], M_app[0], M_app[1], M_app[2]};
            Real eps0[3] = {}, kappa[3] = {};
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    if (i < 3) eps0[i] += abd_inv[i * 6 + j] * rhs[j];
                    else kappa[i - 3] += abd_inv[i * 6 + j] * rhs[j];
                }
            }

            // Find minimum load multiplier for next failure among surviving plies
            Real min_lambda = 1.0e30;
            int fail_ply = -1;
            int fail_mode = 0;

            for (int k = 0; k < lam.num_plies(); ++k) {
                if (ply_failed[k]) continue;

                Real z_mid = (z_bottom[k] + z_top[k]) / 2.0;

                Real Qbar[9];
                composite_detail::compute_Qbar(lam.ply(k), Qbar);

                Real strain_g[3];
                for (int i = 0; i < 3; ++i)
                    strain_g[i] = eps0[i] + z_mid * kappa[i];

                Real stress_g[3];
                for (int i = 0; i < 3; ++i) {
                    stress_g[i] = 0.0;
                    for (int j = 0; j < 3; ++j)
                        stress_g[i] += Qbar[i * 3 + j] * strain_g[j];
                }

                Real theta = lam.ply(k).angle * constants::pi<Real> / 180.0;
                Real stress_l[3];
                composite_detail::transform_to_local(theta, stress_g, stress_l);

                // Hashin indices at unit load - FI scales as lambda^2
                Real s11 = stress_l[0], s22 = stress_l[1], t12 = stress_l[2];

                // Mode 1: Fiber tension
                if (s11 > 0.0) {
                    Real fi = (s11 / Xt_) * (s11 / Xt_) + (t12 / S12_) * (t12 / S12_);
                    if (fi > 1.0e-30) {
                        Real lam_k = 1.0 / std::sqrt(fi);
                        if (lam_k < min_lambda) { min_lambda = lam_k; fail_ply = k; fail_mode = 1; }
                    }
                }
                // Mode 2: Fiber compression
                if (s11 < 0.0) {
                    Real fi = (s11 / Xc_) * (s11 / Xc_);
                    if (fi > 1.0e-30) {
                        Real lam_k = 1.0 / std::sqrt(fi);
                        if (lam_k < min_lambda) { min_lambda = lam_k; fail_ply = k; fail_mode = 2; }
                    }
                }
                // Mode 3: Matrix tension
                if (s22 > 0.0) {
                    Real fi = (s22 / Yt_) * (s22 / Yt_) + (t12 / S12_) * (t12 / S12_);
                    if (fi > 1.0e-30) {
                        Real lam_k = 1.0 / std::sqrt(fi);
                        if (lam_k < min_lambda) { min_lambda = lam_k; fail_ply = k; fail_mode = 3; }
                    }
                }
                // Mode 4: Matrix compression
                if (s22 < 0.0) {
                    Real fi = (s22 / (2.0 * S23_)) * (s22 / (2.0 * S23_))
                             + ((Yc_ / (2.0 * S23_)) * (Yc_ / (2.0 * S23_)) - 1.0) * (s22 / Yc_)
                             + (t12 / S12_) * (t12 / S12_);
                    if (fi > 1.0e-30) {
                        Real lam_k = 1.0 / std::sqrt(fi);
                        if (lam_k < min_lambda) { min_lambda = lam_k; fail_ply = k; fail_mode = 4; }
                    }
                }
            }

            if (fail_ply < 0 || min_lambda > 1.0e20) break;

            cumulative_lambda += min_lambda;
            ply_failed[fail_ply] = true;
            failure_modes[fail_ply] = fail_mode;
            deg_factor[fail_ply] = residual_stiffness_;

            // Recompute degraded ABD after this failure event
            Real post_A[9] = {}, post_B[9] = {}, post_D[9] = {};
            for (int k = 0; k < lam.num_plies(); ++k) {
                if (ply_failed[k] && degradation_model_ == DegradationModel::Ply_Discount)
                    continue;

                Real Qbar_post[9];
                compute_degraded_Qbar(lam.ply(k), ply_failed[k], failure_modes[k],
                                       deg_factor[k], Qbar_post);

                Real zb = z_bottom[k], zt = z_top[k];
                Real dz = zt - zb;
                Real dz2 = zt * zt - zb * zb;
                Real dz3 = zt * zt * zt - zb * zb * zb;

                for (int i = 0; i < 9; ++i) {
                    post_A[i] += Qbar_post[i] * dz;
                    post_B[i] += 0.5 * Qbar_post[i] * dz2;
                    post_D[i] += (1.0 / 3.0) * Qbar_post[i] * dz3;
                }
            }

            // Record result
            auto& res = results[num_events];
            res.load_multiplier = cumulative_lambda;
            res.num_failed_plies = 0;
            for (int k = 0; k < lam.num_plies(); ++k) {
                res.ply_status[k].failed = ply_failed[k];
                res.ply_status[k].failure_mode = failure_modes[k];
                res.ply_status[k].degradation_factor = ply_failed[k] ? residual_stiffness_ : 1.0;
                if (ply_failed[k]) res.num_failed_plies++;
            }
            for (int i = 0; i < 9; ++i) {
                res.degraded_A[i] = post_A[i];
                res.degraded_B[i] = post_B[i];
                res.degraded_D[i] = post_D[i];
            }
            res.laminate_failed = (res.num_failed_plies == lam.num_plies());

            num_events++;
            if (res.laminate_failed) break;
        }

        return num_events;
    }

    /**
     * @brief First-ply-failure analysis using Hashin criterion
     *
     * For quadratic Hashin criteria: FI = a * lambda^2
     * => lambda_FPF = 1 / sqrt(max FI at unit load)
     */
    FPFResult first_ply_failure(const CompositeLaminate& lam,
                                 const Real* N_app, const Real* M_app) const {
        return first_ply_failure_criterion(lam, N_app, M_app, FailureCriterion::Hashin);
    }

    FPFResult first_ply_failure_criterion(const CompositeLaminate& lam,
                                           const Real* N_app, const Real* M_app,
                                           FailureCriterion criterion) const {
        FPFResult result;
        result.criterion = criterion;

        // Solve for unit-load deformation
        Real abd[36];
        lam.get_abd_matrix(abd);
        Real abd_inv[36];
        if (!composite_detail::invert_6x6(abd, abd_inv)) return result;

        Real rhs[6] = {N_app[0], N_app[1], N_app[2], M_app[0], M_app[1], M_app[2]};
        Real eps0[3] = {}, kappa[3] = {};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (i < 3) eps0[i] += abd_inv[i * 6 + j] * rhs[j];
                else kappa[i - 3] += abd_inv[i * 6 + j] * rhs[j];
            }
        }

        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        Real max_fi = 0.0;

        for (int k = 0; k < lam.num_plies(); ++k) {
            Real z_mid = (z_bottom[k] + z_top[k]) / 2.0;

            Real Qbar[9];
            composite_detail::compute_Qbar(lam.ply(k), Qbar);

            Real strain_g[3];
            for (int i = 0; i < 3; ++i)
                strain_g[i] = eps0[i] + z_mid * kappa[i];

            Real stress_g[3];
            for (int i = 0; i < 3; ++i) {
                stress_g[i] = 0.0;
                for (int j = 0; j < 3; ++j)
                    stress_g[i] += Qbar[i * 3 + j] * strain_g[j];
            }

            Real theta = lam.ply(k).angle * constants::pi<Real> / 180.0;
            Real stress_l[3];
            composite_detail::transform_to_local(theta, stress_g, stress_l);

            Real fi = 0.0;
            int mode = 0;

            if (criterion == FailureCriterion::Hashin) {
                fi = compute_hashin_max_fi(stress_l, mode);
            } else {
                fi = compute_tsaiwu_fi(stress_l);
                mode = 1;
            }

            if (fi > max_fi) {
                max_fi = fi;
                result.critical_ply = k;
                result.failure_mode = mode;
            }
        }

        if (max_fi > 1.0e-30) {
            result.load_multiplier = 1.0 / std::sqrt(max_fi);
        }

        return result;
    }

    /**
     * @brief Compute strength envelope in Nxx-Nyy plane
     *
     * Sweeps angle from 0 to 2*pi, computing FPF load multiplier at each direction.
     */
    int strength_envelope(const CompositeLaminate& lam,
                           Real N_ref,
                           StrengthPoint* points,
                           int num_points) const {
        for (int i = 0; i < num_points; ++i) {
            Real angle = 2.0 * constants::pi<Real> * i / num_points;
            Real Nxx = N_ref * std::cos(angle);
            Real Nyy = N_ref * std::sin(angle);

            Real N_app[3] = {Nxx, Nyy, 0.0};
            Real M_app[3] = {0.0, 0.0, 0.0};

            auto fpf = first_ply_failure(lam, N_app, M_app);

            points[i].Nxx = Nxx * fpf.load_multiplier;
            points[i].Nyy = Nyy * fpf.load_multiplier;
            points[i].Nxy = 0.0;
            points[i].critical_ply = fpf.critical_ply;
            points[i].failure_mode = fpf.failure_mode;
        }
        return num_points;
    }

    /**
     * @brief Compute strength envelope in Nxx-Nxy plane
     */
    int strength_envelope_shear(const CompositeLaminate& lam,
                                 Real N_ref,
                                 StrengthPoint* points,
                                 int num_points) const {
        for (int i = 0; i < num_points; ++i) {
            Real angle = 2.0 * constants::pi<Real> * i / num_points;
            Real Nxx = N_ref * std::cos(angle);
            Real Nxy = N_ref * std::sin(angle);

            Real N_app[3] = {Nxx, 0.0, Nxy};
            Real M_app[3] = {0.0, 0.0, 0.0};

            auto fpf = first_ply_failure(lam, N_app, M_app);

            points[i].Nxx = Nxx * fpf.load_multiplier;
            points[i].Nyy = 0.0;
            points[i].Nxy = Nxy * fpf.load_multiplier;
            points[i].critical_ply = fpf.critical_ply;
            points[i].failure_mode = fpf.failure_mode;
        }
        return num_points;
    }

private:
    Real Xt_, Xc_, Yt_, Yc_, S12_, S23_;
    DegradationModel degradation_model_;
    Real residual_stiffness_;
    Real F12_star_;

    Real compute_hashin_max_fi(const Real* stress_l, int& mode) const {
        Real s11 = stress_l[0], s22 = stress_l[1], t12 = stress_l[2];
        Real max_fi = 0.0;
        mode = 0;

        if (s11 > 0.0) {
            Real f1 = (s11 / Xt_) * (s11 / Xt_) + (t12 / S12_) * (t12 / S12_);
            if (f1 > max_fi) { max_fi = f1; mode = 1; }
        }
        if (s11 < 0.0) {
            Real f2 = (s11 / Xc_) * (s11 / Xc_);
            if (f2 > max_fi) { max_fi = f2; mode = 2; }
        }
        if (s22 > 0.0) {
            Real f3 = (s22 / Yt_) * (s22 / Yt_) + (t12 / S12_) * (t12 / S12_);
            if (f3 > max_fi) { max_fi = f3; mode = 3; }
        }
        if (s22 < 0.0) {
            Real f4 = (s22 / (2.0 * S23_)) * (s22 / (2.0 * S23_))
                     + ((Yc_ / (2.0 * S23_)) * (Yc_ / (2.0 * S23_)) - 1.0) * (s22 / Yc_)
                     + (t12 / S12_) * (t12 / S12_);
            if (f4 > max_fi) { max_fi = f4; mode = 4; }
        }
        return max_fi;
    }

    Real compute_tsaiwu_fi(const Real* stress_l) const {
        Real s1 = stress_l[0], s2 = stress_l[1], t12 = stress_l[2];

        Real F1 = 1.0 / Xt_ - 1.0 / Xc_;
        Real F2 = 1.0 / Yt_ - 1.0 / Yc_;
        Real F11 = 1.0 / (Xt_ * Xc_);
        Real F22 = 1.0 / (Yt_ * Yc_);
        Real F66 = 1.0 / (S12_ * S12_);
        Real F12 = F12_star_ * std::sqrt(F11 * F22);

        // Tsai-Wu: F = a*lambda^2 + b*lambda where
        // a = F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2
        // b = F1*s1 + F2*s2
        // We want F(lambda) = 1 => a*lam^2 + b*lam - 1 = 0
        // For the equivalent "failure index at unit load" approach,
        // we solve the quadratic and return 1/lambda^2 as the FI analog
        Real a = F11 * s1 * s1 + F22 * s2 * s2 + F66 * t12 * t12 + 2.0 * F12 * s1 * s2;
        Real b = F1 * s1 + F2 * s2;

        // Solve a*lam^2 + b*lam - 1 = 0
        if (std::fabs(a) < 1.0e-30) {
            // Linear case
            if (std::fabs(b) < 1.0e-30) return 0.0;
            Real lam_val = 1.0 / b;
            if (lam_val < 0.0) return 0.0;
            return 1.0 / (lam_val * lam_val);
        }

        Real disc = b * b + 4.0 * a;
        if (disc < 0.0) return 0.0;

        Real lam_val = (-b + std::sqrt(disc)) / (2.0 * a);
        if (lam_val <= 0.0) return 0.0;
        return 1.0 / (lam_val * lam_val);
    }

    void compute_degraded_Qbar(const PlyDefinition& ply, bool failed, int fail_mode,
                                 Real deg_factor, Real* Qbar) const {
        composite_detail::compute_Qbar(ply, Qbar);

        if (!failed) return;

        if (degradation_model_ == DegradationModel::Ply_Discount) {
            for (int i = 0; i < 9; ++i) Qbar[i] *= residual_stiffness_;
        } else if (degradation_model_ == DegradationModel::Selective_Discount) {
            // Fiber failure (modes 1,2): degrade fiber-direction stiffness
            if (fail_mode == 1 || fail_mode == 2) {
                // Degrade Q11-related terms
                Qbar[0] *= residual_stiffness_;  // Q_bar_11
                Qbar[2] *= residual_stiffness_;  // Q_bar_16
                Qbar[6] *= residual_stiffness_;  // Q_bar_61
            }
            // Matrix failure (modes 3,4): degrade transverse stiffness
            if (fail_mode == 3 || fail_mode == 4) {
                Qbar[4] *= residual_stiffness_;  // Q_bar_22
                Qbar[5] *= residual_stiffness_;  // Q_bar_26
                Qbar[7] *= residual_stiffness_;  // Q_bar_62
                Qbar[8] *= residual_stiffness_;  // Q_bar_66
            }
        } else {
            // Gradual
            for (int i = 0; i < 9; ++i) Qbar[i] *= deg_factor;
        }
    }

    void compute_degraded_abd(const CompositeLaminate& lam,
                                ProgressiveFailureResult& result) const {
        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        for (int i = 0; i < 9; ++i) {
            result.degraded_A[i] = 0.0;
            result.degraded_B[i] = 0.0;
            result.degraded_D[i] = 0.0;
        }

        for (int k = 0; k < lam.num_plies(); ++k) {
            Real Qbar[9];
            compute_degraded_Qbar(lam.ply(k),
                                    result.ply_status[k].failed,
                                    result.ply_status[k].failure_mode,
                                    result.ply_status[k].degradation_factor,
                                    Qbar);

            Real zb = z_bottom[k], zt = z_top[k];
            Real dz = zt - zb;
            Real dz2 = zt * zt - zb * zb;
            Real dz3 = zt * zt * zt - zb * zb * zb;

            for (int i = 0; i < 9; ++i) {
                result.degraded_A[i] += Qbar[i] * dz;
                result.degraded_B[i] += 0.5 * Qbar[i] * dz2;
                result.degraded_D[i] += (1.0 / 3.0) * Qbar[i] * dz3;
            }
        }
    }
};

} // namespace physics
} // namespace nxs
