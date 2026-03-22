#pragma once

/**
 * @file acoustic_wave37.hpp
 * @brief Wave 37: Acoustic Analysis (4 features)
 *
 * Features:
 *   7. NoiseComputation     - Structural noise from surface velocity
 *   8. AcousticPressure     - Kirchhoff integral (monopole + dipole)
 *   9. NoiseFilter          - Octave band analysis and A-weighting
 *  10. AcousticBEM          - Boundary element method for exterior acoustics
 *
 * Namespace: nxs::physics
 *
 * References:
 *  - Fahy & Gardonio (2007), "Sound and Structural Vibration", 2nd ed.
 *  - Marburg & Nolte (2008), "Computational Acoustics of Noise Propagation"
 *  - ISO 226:2003, "Normal Equal-Loudness-Level Contours" (A-weighting)
 *  - Kirkup (2007), "The Boundary Element Method in Acoustics"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <complex>

namespace nxs {
namespace physics {

using Real = nxs::Real;

// ============================================================================
// Constants
// ============================================================================

namespace acoustic_constants {
    static constexpr Real PI = 3.14159265358979323846;
    static constexpr Real TWO_PI = 2.0 * PI;
    static constexpr Real FOUR_PI = 4.0 * PI;
    static constexpr Real REF_POWER = 1.0e-12;        ///< Reference sound power (W)
    static constexpr Real REF_PRESSURE = 2.0e-5;      ///< Reference sound pressure (Pa)
    static constexpr Real DEFAULT_RHO = 1.225;         ///< Air density (kg/m^3)
    static constexpr Real DEFAULT_C = 343.0;           ///< Speed of sound in air (m/s)
} // namespace acoustic_constants


// ============================================================================
// 7. NoiseComputation — Structural noise from surface velocity
// ============================================================================

/**
 * @brief Surface panel for noise computation
 */
struct SurfacePanel {
    Real area;           ///< Panel area (m^2)
    Real normal[3];      ///< Outward unit normal
    Real velocity[3];    ///< Surface velocity vector (m/s)
    Real center[3];      ///< Panel centroid coordinates (m)
};

/**
 * @brief Structural noise power computation
 *
 * Computes radiated sound power from vibrating surface panels using
 * the Rayleigh integral approximation:
 *
 *   W = sum_i (rho * c * v_n_i^2 * A_i)
 *
 * where v_n_i is the normal velocity component on panel i, and A_i is
 * the panel area. This is valid for a baffled planar source in the
 * far-field limit.
 *
 * Sound power level:  Lw = 10 * log10(W / W_ref), W_ref = 1e-12 W
 */
class NoiseComputation {
public:
    NoiseComputation(Real rho = acoustic_constants::DEFAULT_RHO,
                     Real c = acoustic_constants::DEFAULT_C)
        : rho_(rho), c_(c) {}

    /// Compute the normal velocity component of a panel (v dot n)
    static Real normal_velocity(const SurfacePanel& panel) {
        return panel.velocity[0] * panel.normal[0] +
               panel.velocity[1] * panel.normal[1] +
               panel.velocity[2] * panel.normal[2];
    }

    /// Compute total radiated sound power from all panels (Watts)
    Real compute_noise_power(const SurfacePanel* panels, int n_panels) const {
        Real total_power = 0.0;
        for (int i = 0; i < n_panels; ++i) {
            Real vn = normal_velocity(panels[i]);
            total_power += rho_ * c_ * vn * vn * panels[i].area;
        }
        return total_power;
    }

    /// Compute sound power level in dB (re 1e-12 W)
    Real compute_power_level_dB(const SurfacePanel* panels, int n_panels) const {
        Real W = compute_noise_power(panels, n_panels);
        if (W <= 0.0) return -std::numeric_limits<Real>::infinity();
        return 10.0 * std::log10(W / acoustic_constants::REF_POWER);
    }

    /// Compute sound power contribution per panel (for ranking)
    void compute_panel_powers(const SurfacePanel* panels, int n_panels,
                              Real* powers) const {
        for (int i = 0; i < n_panels; ++i) {
            Real vn = normal_velocity(panels[i]);
            powers[i] = rho_ * c_ * vn * vn * panels[i].area;
        }
    }

    /// Compute mean-square normal velocity over the surface
    static Real mean_square_velocity(const SurfacePanel* panels, int n_panels) {
        Real sum_v2A = 0.0;
        Real sum_A = 0.0;
        for (int i = 0; i < n_panels; ++i) {
            Real vn = normal_velocity(panels[i]);
            sum_v2A += vn * vn * panels[i].area;
            sum_A += panels[i].area;
        }
        return (sum_A > 0.0) ? (sum_v2A / sum_A) : 0.0;
    }

    /// Compute radiation efficiency: sigma = W / (rho * c * <v^2> * S_total)
    Real radiation_efficiency(const SurfacePanel* panels, int n_panels) const {
        Real W = compute_noise_power(panels, n_panels);
        Real v2_mean = mean_square_velocity(panels, n_panels);
        Real S_total = 0.0;
        for (int i = 0; i < n_panels; ++i) S_total += panels[i].area;
        Real denom = rho_ * c_ * v2_mean * S_total;
        if (denom <= 0.0) return 0.0;
        return W / denom;
    }

    Real rho() const { return rho_; }
    Real speed_of_sound() const { return c_; }

private:
    Real rho_;
    Real c_;
};


// ============================================================================
// 8. AcousticPressure — Kirchhoff integral (far-field monopole + dipole)
// ============================================================================

/**
 * @brief Kirchhoff integral for acoustic pressure computation
 *
 * Computes the acoustic pressure at an observer point from a vibrating
 * surface using the Kirchhoff-Helmholtz integral equation:
 *
 * p(x) = integral_S [ ik*rho*c * v_n * G(x,y) + p_s * dG/dn ] dS
 *
 * In the simplified far-field model (monopole + dipole):
 *
 * Monopole:  p_m = (ik * rho * c / 4pi) * sum_i v_n_i * A_i * exp(-ikr_i) / r_i
 * Dipole:    p_d = (1 / 4pi) * sum_i v_n_i * A_i * (ikr_i + 1) * cos(theta_i) *
 *                  exp(-ikr_i) / r_i^2
 *
 * where k = 2*pi*f/c is the wavenumber, r_i is the distance from panel i
 * to the observer, and theta_i is the angle between the panel normal and
 * the direction to the observer.
 */
class AcousticPressure {
public:
    AcousticPressure(Real rho = acoustic_constants::DEFAULT_RHO,
                     Real c = acoustic_constants::DEFAULT_C)
        : rho_(rho), c_(c) {}

    using Complex = std::complex<Real>;

    /// Compute distance between two 3D points
    static Real distance(const Real* a, const Real* b) {
        Real dx = a[0] - b[0];
        Real dy = a[1] - b[1];
        Real dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// Compute unit direction vector from a to b, return distance
    static Real direction(const Real* from, const Real* to, Real* dir) {
        dir[0] = to[0] - from[0];
        dir[1] = to[1] - from[1];
        dir[2] = to[2] - from[2];
        Real r = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (r > 1.0e-30) {
            dir[0] /= r; dir[1] /= r; dir[2] /= r;
        }
        return r;
    }

    /// Dot product of two 3-vectors
    static Real dot3(const Real* a, const Real* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    /// Compute monopole pressure contribution from all panels at observer
    Complex compute_monopole(const SurfacePanel* panels, int n_panels,
                             const Real* observer, Real frequency) const {
        Real k = acoustic_constants::TWO_PI * frequency / c_;
        Complex ik(0.0, k);
        Complex coeff = ik * rho_ * c_ / acoustic_constants::FOUR_PI;

        Complex p_total(0.0, 0.0);
        for (int i = 0; i < n_panels; ++i) {
            Real vn = NoiseComputation::normal_velocity(panels[i]);
            Real r = distance(panels[i].center, observer);
            if (r < 1.0e-15) continue;
            Complex G = std::exp(-ik * r) / r;
            p_total += coeff * vn * panels[i].area * G;
        }
        return p_total;
    }

    /// Compute dipole pressure contribution from all panels at observer
    Complex compute_dipole(const SurfacePanel* panels, int n_panels,
                           const Real* observer, Real frequency) const {
        Real k = acoustic_constants::TWO_PI * frequency / c_;
        Complex ik(0.0, k);

        Complex p_total(0.0, 0.0);
        for (int i = 0; i < n_panels; ++i) {
            Real vn = NoiseComputation::normal_velocity(panels[i]);
            Real dir[3];
            Real r = direction(panels[i].center, observer, dir);
            if (r < 1.0e-15) continue;

            Real cos_theta = dot3(panels[i].normal, dir);

            // dG/dn = cos(theta) * (ik*r + 1) * exp(-ikr) / (4*pi*r^2)
            // but without the 4pi factor (added outside)
            Complex exp_ikr = std::exp(-ik * r);
            Complex dGdn = cos_theta * (ik * r + 1.0) * exp_ikr / (r * r);

            // Dipole contribution (sign convention: outward normal)
            p_total += vn * panels[i].area * dGdn / acoustic_constants::FOUR_PI;
        }
        return p_total;
    }

    /// Compute total acoustic pressure (monopole + dipole) at observer
    /// Returns the complex pressure amplitude
    Complex compute_pressure(const SurfacePanel* panels, int n_panels,
                             const Real* observer, Real frequency) const {
        return compute_monopole(panels, n_panels, observer, frequency) +
               compute_dipole(panels, n_panels, observer, frequency);
    }

    /// Compute pressure magnitude (Pa)
    Real compute_pressure_magnitude(const SurfacePanel* panels, int n_panels,
                                    const Real* observer, Real frequency) const {
        Complex p = compute_pressure(panels, n_panels, observer, frequency);
        return std::abs(p);
    }

    /// Compute sound pressure level in dB (re 20 uPa)
    Real compute_spl_dB(const SurfacePanel* panels, int n_panels,
                        const Real* observer, Real frequency) const {
        Real p_mag = compute_pressure_magnitude(panels, n_panels, observer, frequency);
        if (p_mag <= 0.0) return -std::numeric_limits<Real>::infinity();
        return 20.0 * std::log10(p_mag / acoustic_constants::REF_PRESSURE);
    }

    /// Compute pressure field on a grid of observers
    void compute_pressure_field(const SurfacePanel* panels, int n_panels,
                                const Real* observers, int n_obs,
                                Real frequency, Real* p_magnitude) const {
        for (int i = 0; i < n_obs; ++i) {
            Complex p = compute_pressure(panels, n_panels,
                                         &observers[3 * i], frequency);
            p_magnitude[i] = std::abs(p);
        }
    }

    Real rho() const { return rho_; }
    Real speed_of_sound() const { return c_; }

private:
    Real rho_;
    Real c_;
};


// ============================================================================
// 9. NoiseFilter — Octave band analysis and A-weighting
// ============================================================================

/**
 * @brief Octave band data
 */
struct OctaveBand {
    Real center_freq;   ///< Band center frequency (Hz)
    Real lower;         ///< Lower band edge (Hz)
    Real upper;         ///< Upper band edge (Hz)
    Real level_dB;      ///< Sound level in this band (dB)
};

/**
 * @brief Noise filter: octave band decomposition and A-weighting
 *
 * Standard octave bands (base-10): f_c = 10^(n/10) * 1000 Hz
 * 1/3 octave bands: f_c = 10^(n/30) * 1000 Hz
 * Band edges: f_lower = f_c / 2^(1/(2*N)), f_upper = f_c * 2^(1/(2*N))
 * where N=1 for octave, N=3 for 1/3 octave.
 *
 * A-weighting per IEC 61672-1:
 *   R_A(f) = 12194^2 * f^4 / [(f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)*(f^2 + 737.9^2)) * (f^2 + 12194^2)]
 *   A(f) = 20*log10(R_A(f)) + 2.0  (offset to 0 dB at 1 kHz)
 */
class NoiseFilter {
public:
    /// Generate standard 1/1 octave band center frequencies (31.5 Hz to 16 kHz)
    static void generate_octave_centers(std::vector<Real>& centers) {
        // Standard preferred frequencies: 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k
        static const Real std_centers[] = {
            31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0
        };
        centers.assign(std::begin(std_centers), std::end(std_centers));
    }

    /// Generate 1/3 octave band center frequencies from f_min to f_max
    /// f_c = 10^(n/30) * 1000 Hz for integer n
    static void generate_third_octave_centers(Real f_min, Real f_max,
                                              std::vector<Real>& centers) {
        centers.clear();
        // n ranges typically from -16 (25 Hz) to +13 (20 kHz)
        for (int n = -20; n <= 15; ++n) {
            Real fc = 1000.0 * std::pow(10.0, static_cast<Real>(n) / 30.0);
            if (fc >= f_min && fc <= f_max) {
                centers.push_back(fc);
            }
        }
    }

    /// Compute octave band edges from center frequency (1/N octave)
    static void compute_band_edges(Real f_center, int N, Real& f_lower, Real& f_upper) {
        Real factor = std::pow(2.0, 1.0 / (2.0 * N));
        f_lower = f_center / factor;
        f_upper = f_center * factor;
    }

    /// Decompose a frequency spectrum into octave bands
    /// spectrum: array of (frequency_Hz, level_dB) pairs, n_freq entries
    /// Energetically sums contributions in each band
    void compute_octave_bands(const Real* freq, const Real* level_dB, int n_freq,
                              OctaveBand* bands, int n_bands) const {
        for (int b = 0; b < n_bands; ++b) {
            Real f_lo = bands[b].lower;
            Real f_hi = bands[b].upper;

            // Sum energy in this band: E = sum 10^(L_i/10)
            Real energy = 0.0;
            for (int i = 0; i < n_freq; ++i) {
                if (freq[i] >= f_lo && freq[i] < f_hi) {
                    energy += std::pow(10.0, level_dB[i] / 10.0);
                }
            }
            bands[b].level_dB = (energy > 0.0) ? 10.0 * std::log10(energy) : -200.0;
        }
    }

    /// A-weighting correction for a given frequency (dB)
    /// Based on IEC 61672-1 formula
    static Real a_weighting_correction(Real f) {
        if (f <= 0.0) return -200.0;

        Real f2 = f * f;
        Real f4 = f2 * f2;

        // Numerator: 12194^2 * f^4
        static constexpr Real c1 = 12194.0;
        static constexpr Real c2 = 20.6;
        static constexpr Real c3 = 107.7;
        static constexpr Real c4 = 737.9;

        Real c1_2 = c1 * c1;
        Real c2_2 = c2 * c2;
        Real c3_2 = c3 * c3;
        Real c4_2 = c4 * c4;

        Real num = c1_2 * f4;
        Real denom = (f2 + c2_2) *
                     std::sqrt((f2 + c3_2) * (f2 + c4_2)) *
                     (f2 + c1_2);

        if (denom <= 0.0) return -200.0;

        Real R_A = num / denom;
        // The offset +2.0 normalizes to 0 dB at 1 kHz
        return 20.0 * std::log10(R_A) + 2.0;
    }

    /// Apply A-weighting to all octave bands
    static void apply_a_weighting(OctaveBand* bands, int n_bands) {
        for (int i = 0; i < n_bands; ++i) {
            Real correction = a_weighting_correction(bands[i].center_freq);
            bands[i].level_dB += correction;
        }
    }

    /// Compute overall A-weighted level from individual band levels
    static Real overall_a_weighted_level(const OctaveBand* bands, int n_bands) {
        Real total_energy = 0.0;
        for (int i = 0; i < n_bands; ++i) {
            total_energy += std::pow(10.0, bands[i].level_dB / 10.0);
        }
        return (total_energy > 0.0) ? 10.0 * std::log10(total_energy) : -200.0;
    }

    /// Compute C-weighting correction (flatter than A-weighting)
    static Real c_weighting_correction(Real f) {
        if (f <= 0.0) return -200.0;

        Real f2 = f * f;
        static constexpr Real c1 = 12194.0;
        static constexpr Real c2 = 20.6;
        Real c1_2 = c1 * c1;
        Real c2_2 = c2 * c2;

        Real num = c1_2 * f2;
        Real denom = (f2 + c2_2) * (f2 + c1_2);

        if (denom <= 0.0) return -200.0;
        Real R_C = num / denom;
        return 20.0 * std::log10(R_C) + 0.062;
    }
};


// ============================================================================
// 10. AcousticBEM — Boundary Element Method for exterior acoustics
// ============================================================================

/**
 * @brief Boundary element panel for acoustic BEM
 */
struct BEMPanel {
    Real center[3];   ///< Panel centroid
    Real normal[3];   ///< Outward unit normal
    Real area;        ///< Panel area (m^2)
};

/**
 * @brief Boundary Element Method for exterior acoustic problems
 *
 * Solves the Helmholtz exterior boundary value problem using
 * collocation BEM with constant elements.
 *
 * The Kirchhoff-Helmholtz integral equation:
 *   c(x) * p(x) = integral_S [ G(x,y) * dp/dn(y) - dG/dn(x,y) * p(y) ] dS(y)
 *
 * where c(x) = 0.5 for smooth boundary points, and:
 *   G(x,y) = exp(-ik|x-y|) / (4*pi*|x-y|)      (free-space Green's function)
 *   dG/dn  = (ik|r| + 1) * cos(theta) * G / |r|  (normal derivative)
 *
 * For Neumann BC (given v_n -> dp/dn = -i*omega*rho*v_n):
 *   [H] * {p} = [G] * {q}
 *
 * where H_ij = c_i*delta_ij + integral dG/dn dS_j
 *       G_ij = integral G dS_j
 *       q_j = -i*omega*rho*v_n_j
 *
 * With constant elements, the integrals reduce to G(x_i, x_j) * A_j.
 */
class AcousticBEM {
public:
    using Complex = std::complex<Real>;

    AcousticBEM(Real rho = acoustic_constants::DEFAULT_RHO,
                Real c = acoustic_constants::DEFAULT_C)
        : rho_(rho), c_(c) {}

    /// Free-space Green's function: G = exp(-ikr) / (4*pi*r)
    Complex greens_function(Real r, Real k) const {
        if (r < 1.0e-15) return Complex(0.0, 0.0);
        Complex ikr(0.0, k * r);
        return std::exp(-ikr) / (acoustic_constants::FOUR_PI * r);
    }

    /// Normal derivative of Green's function
    /// dG/dn = [(ikr + 1) / r] * cos(theta) * G
    Complex greens_normal_derivative(Real r, Real cos_theta, Real k) const {
        if (r < 1.0e-15) return Complex(0.0, 0.0);
        Complex ikr(0.0, k * r);
        Complex G = std::exp(-ikr) / (acoustic_constants::FOUR_PI * r);
        return (ikr + 1.0) * cos_theta * G / r;
    }

    /// Compute distance between two 3D points
    static Real dist(const Real* a, const Real* b) {
        Real dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// Dot product
    static Real dot3(const Real* a, const Real* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    /// Assemble BEM matrices H and G for n panels at given frequency
    /// H_mat, G_mat: row-major n x n complex arrays (caller allocates)
    void assemble_bem(const BEMPanel* panels, int n, Real freq,
                      Complex* H_mat, Complex* G_mat) const {
        Real k = acoustic_constants::TWO_PI * freq / c_;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    // Diagonal: c_i = 0.5 for smooth boundary
                    H_mat[i * n + j] = Complex(0.5, 0.0);
                    // Self-integral of G for constant element:
                    // approximate as G(r_eff) * A with r_eff = sqrt(A/pi)
                    Real r_eff = std::sqrt(panels[j].area / acoustic_constants::PI);
                    G_mat[i * n + j] = greens_function(r_eff, k) * panels[j].area;
                } else {
                    Real r = dist(panels[i].center, panels[j].center);

                    // Direction from j to i
                    Real dir[3];
                    dir[0] = panels[i].center[0] - panels[j].center[0];
                    dir[1] = panels[i].center[1] - panels[j].center[1];
                    dir[2] = panels[i].center[2] - panels[j].center[2];
                    if (r > 1.0e-15) {
                        dir[0] /= r; dir[1] /= r; dir[2] /= r;
                    }

                    Real cos_theta = dot3(panels[j].normal, dir);

                    G_mat[i * n + j] = greens_function(r, k) * panels[j].area;
                    H_mat[i * n + j] = greens_normal_derivative(r, cos_theta, k)
                                       * panels[j].area;
                }
            }
        }
    }

    /// Solve BEM system: H * p = G * q, where q_j = -i*omega*rho*v_n_j
    /// Uses simple Gauss elimination for small systems
    /// v_n: normal velocity on each panel, pressure: output surface pressure
    void solve_bem(const BEMPanel* panels, int n, Real freq,
                   const Real* v_n, Complex* pressure) const {
        Real omega = acoustic_constants::TWO_PI * freq;
        Complex i_omega_rho(0.0, -omega * rho_);

        int n2 = n * n;
        std::vector<Complex> H(n2), G(n2);
        assemble_bem(panels, n, freq, H.data(), G.data());

        // Compute RHS: b = G * q
        std::vector<Complex> rhs(n);
        for (int i = 0; i < n; ++i) {
            rhs[i] = Complex(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                Complex q_j = i_omega_rho * v_n[j];
                rhs[i] += G[i * n + j] * q_j;
            }
        }

        // Solve H * p = rhs via Gauss elimination with partial pivoting
        // Augmented matrix: [H | rhs]
        std::vector<Complex> aug(n * (n + 1));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                aug[i * (n + 1) + j] = H[i * n + j];
            }
            aug[i * (n + 1) + n] = rhs[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; ++col) {
            // Find pivot
            Real best_mag = 0.0;
            int pivot_row = col;
            for (int row = col; row < n; ++row) {
                Real mag = std::abs(aug[row * (n + 1) + col]);
                if (mag > best_mag) {
                    best_mag = mag;
                    pivot_row = row;
                }
            }
            // Swap rows
            if (pivot_row != col) {
                for (int j = 0; j <= n; ++j) {
                    std::swap(aug[col * (n + 1) + j], aug[pivot_row * (n + 1) + j]);
                }
            }
            // Eliminate
            Complex pivot = aug[col * (n + 1) + col];
            if (std::abs(pivot) < 1.0e-30) continue;

            for (int row = col + 1; row < n; ++row) {
                Complex factor = aug[row * (n + 1) + col] / pivot;
                for (int j = col; j <= n; ++j) {
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
                }
            }
        }

        // Back substitution
        for (int i = n - 1; i >= 0; --i) {
            Complex sum = aug[i * (n + 1) + n];
            for (int j = i + 1; j < n; ++j) {
                sum -= aug[i * (n + 1) + j] * pressure[j];
            }
            Complex diag = aug[i * (n + 1) + i];
            pressure[i] = (std::abs(diag) > 1.0e-30) ? (sum / diag) : Complex(0.0, 0.0);
        }
    }

    /// Compute pressure at an exterior field point from BEM surface solution
    Complex field_pressure(const BEMPanel* panels, int n, Real freq,
                           const Real* v_n, const Complex* surface_p,
                           const Real* field_pt) const {
        Real k = acoustic_constants::TWO_PI * freq / c_;
        Real omega = acoustic_constants::TWO_PI * freq;
        Complex i_omega_rho(0.0, -omega * rho_);

        Complex p_field(0.0, 0.0);
        for (int j = 0; j < n; ++j) {
            Real r = dist(field_pt, panels[j].center);
            if (r < 1.0e-15) continue;

            Real dir[3];
            dir[0] = field_pt[0] - panels[j].center[0];
            dir[1] = field_pt[1] - panels[j].center[1];
            dir[2] = field_pt[2] - panels[j].center[2];
            dir[0] /= r; dir[1] /= r; dir[2] /= r;

            Real cos_theta = dot3(panels[j].normal, dir);

            Complex G_val = greens_function(r, k) * panels[j].area;
            Complex dGdn = greens_normal_derivative(r, cos_theta, k) * panels[j].area;

            Complex q_j = i_omega_rho * v_n[j];
            p_field += G_val * q_j - dGdn * surface_p[j];
        }
        return p_field;
    }

    /// Compute the wavenumber for a given frequency
    Real wavenumber(Real freq) const {
        return acoustic_constants::TWO_PI * freq / c_;
    }

    Real rho() const { return rho_; }
    Real speed_of_sound() const { return c_; }

private:
    Real rho_;
    Real c_;
};

} // namespace physics
} // namespace nxs
