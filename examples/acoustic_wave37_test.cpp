/**
 * @file acoustic_wave37_test.cpp
 * @brief Wave 37: Acoustic Analysis Test Suite (4 features, 30 tests)
 *
 * Tests:
 *   7.  NoiseComputation   (8 tests)
 *   8.  AcousticPressure   (8 tests)
 *   9.  NoiseFilter        (7 tests)
 *   10. AcousticBEM        (7 tests)
 */

#include <nexussim/physics/acoustic_wave37.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

using namespace nxs;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)


// ============================================================================
// 7. NoiseComputation Tests
// ============================================================================

void test_7_noise_computation() {
    std::cout << "--- Test 7: NoiseComputation ---\n";

    Real rho = 1.225;
    Real c = 343.0;
    NoiseComputation noise(rho, c);

    // 7a. Single panel: uniform normal velocity
    // Panel area=1 m^2, normal=(0,0,1), velocity=(0,0,1) m/s
    // W = rho*c * vn^2 * A = 1.225*343*1*1 = 420.175 W
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real W = noise.compute_noise_power(&panel, 1);
        CHECK_NEAR(W, rho * c, 1e-6, "Noise: single panel power = rho*c");
    }

    // 7b. Power scales with velocity squared
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 2.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real W = noise.compute_noise_power(&panel, 1);
        CHECK_NEAR(W, rho * c * 4.0, 1e-6, "Noise: power scales v^2");
    }

    // 7c. Power scales with area
    {
        SurfacePanel panel;
        panel.area = 3.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real W = noise.compute_noise_power(&panel, 1);
        CHECK_NEAR(W, rho * c * 3.0, 1e-6, "Noise: power scales with area");
    }

    // 7d. Tangential velocity contributes zero
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 5.0; panel.velocity[1] = 3.0; panel.velocity[2] = 0.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real W = noise.compute_noise_power(&panel, 1);
        CHECK_NEAR(W, 0.0, 1e-15, "Noise: tangential velocity => 0 power");
    }

    // 7e. Sound power level in dB
    // W = rho*c = 420.175 W, Lw = 10*log10(420.175/1e-12)
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real Lw = noise.compute_power_level_dB(&panel, 1);
        Real expected = 10.0 * std::log10(rho * c / 1e-12);
        CHECK_NEAR(Lw, expected, 0.01, "Noise: power level dB");
    }

    // 7f. Multiple panels: powers sum
    {
        SurfacePanel panels[2];
        for (int i = 0; i < 2; ++i) {
            panels[i].area = 1.0;
            panels[i].normal[0] = 0; panels[i].normal[1] = 0; panels[i].normal[2] = 1;
            panels[i].velocity[0] = 0; panels[i].velocity[1] = 0; panels[i].velocity[2] = 1.0;
            panels[i].center[0] = static_cast<Real>(i); panels[i].center[1] = 0; panels[i].center[2] = 0;
        }
        Real W = noise.compute_noise_power(panels, 2);
        CHECK_NEAR(W, 2.0 * rho * c, 1e-6, "Noise: 2 panels sum");
    }

    // 7g. Mean square velocity
    {
        SurfacePanel panel;
        panel.area = 2.0;
        panel.normal[0] = 1; panel.normal[1] = 0; panel.normal[2] = 0;
        panel.velocity[0] = 3.0; panel.velocity[1] = 0; panel.velocity[2] = 0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real msv = NoiseComputation::mean_square_velocity(&panel, 1);
        CHECK_NEAR(msv, 9.0, 1e-15, "Noise: mean square velocity = 9");
    }

    // 7h. Radiation efficiency for baffled flat plate = 1.0
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real sigma = noise.radiation_efficiency(&panel, 1);
        CHECK_NEAR(sigma, 1.0, 1e-10, "Noise: radiation efficiency = 1 for flat plate");
    }
}


// ============================================================================
// 8. AcousticPressure Tests
// ============================================================================

void test_8_acoustic_pressure() {
    std::cout << "\n--- Test 8: AcousticPressure ---\n";

    Real rho = 1.225;
    Real c_sound = 343.0;
    AcousticPressure ap(rho, c_sound);

    // 8a. Distance computation
    {
        Real a[3] = {0, 0, 0};
        Real b[3] = {3, 4, 0};
        Real d = AcousticPressure::distance(a, b);
        CHECK_NEAR(d, 5.0, 1e-12, "AcPressure: distance 3-4-5");
    }

    // 8b. Monopole from single panel: far-field amplitude ~ (rho*c*k*vn*A)/(4*pi*r)
    // f=1000 Hz, k=2*pi*1000/343 ~ 18.31, vn=1, A=0.01, r=10
    // |p_monopole| ~ rho*c*k*A/(4*pi*r) * |vn| = 1.225*343*18.31*0.01/(4*pi*10)
    {
        SurfacePanel panel;
        panel.area = 0.01;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real observer[3] = {0, 0, 10.0};
        Real freq = 1000.0;

        auto p_m = ap.compute_monopole(&panel, 1, observer, freq);
        Real k = 2.0 * acoustic_constants::PI * freq / c_sound;
        Real expected_mag = rho * c_sound * k * 0.01 / (4.0 * acoustic_constants::PI * 10.0);
        CHECK_NEAR(std::abs(p_m), expected_mag, expected_mag * 0.01,
                   "AcPressure: monopole magnitude");
    }

    // 8c. Pressure decays with distance (1/r for monopole)
    {
        SurfacePanel panel;
        panel.area = 0.01;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real obs_near[3] = {0, 0, 5.0};
        Real obs_far[3] = {0, 0, 10.0};
        Real freq = 500.0;

        auto p_near = ap.compute_monopole(&panel, 1, obs_near, freq);
        auto p_far = ap.compute_monopole(&panel, 1, obs_far, freq);

        Real ratio = std::abs(p_near) / std::abs(p_far);
        CHECK_NEAR(ratio, 2.0, 0.01, "AcPressure: 1/r decay monopole");
    }

    // 8d. SPL is finite and positive for non-zero source
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real observer[3] = {0, 0, 1.0};
        Real spl = ap.compute_spl_dB(&panel, 1, observer, 1000.0);
        CHECK(spl > 0.0, "AcPressure: SPL positive for non-zero source");
        CHECK(std::isfinite(spl), "AcPressure: SPL is finite");
    }

    // 8e. Zero velocity => zero pressure
    {
        SurfacePanel panel;
        panel.area = 1.0;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 0.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real observer[3] = {0, 0, 5.0};
        Real p = ap.compute_pressure_magnitude(&panel, 1, observer, 1000.0);
        CHECK_NEAR(p, 0.0, 1e-15, "AcPressure: zero velocity => zero pressure");
    }

    // 8f. Total pressure (monopole + dipole) is non-zero
    {
        SurfacePanel panel;
        panel.area = 0.1;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real observer[3] = {5.0, 0, 0};
        auto p = ap.compute_pressure(&panel, 1, observer, 500.0);
        CHECK(std::abs(p) > 0.0, "AcPressure: total pressure nonzero");
    }

    // 8g. Pressure field computation
    {
        SurfacePanel panel;
        panel.area = 0.01;
        panel.normal[0] = 0; panel.normal[1] = 0; panel.normal[2] = 1;
        panel.velocity[0] = 0; panel.velocity[1] = 0; panel.velocity[2] = 1.0;
        panel.center[0] = 0; panel.center[1] = 0; panel.center[2] = 0;

        Real observers[6] = {0,0,5, 0,0,10};
        Real mags[2] = {};
        ap.compute_pressure_field(&panel, 1, observers, 2, 1000.0, mags);
        CHECK(mags[0] > mags[1], "AcPressure: closer observer louder");
    }

    // 8h. Wavenumber k = 2*pi*f/c
    {
        Real a[3] = {1.0, 2.0, 3.0};
        Real b[3] = {4.0, 5.0, 6.0};
        Real d = AcousticPressure::dot3(a, b);
        CHECK_NEAR(d, 32.0, 1e-12, "AcPressure: dot3 = 32");
    }
}


// ============================================================================
// 9. NoiseFilter Tests
// ============================================================================

void test_9_noise_filter() {
    std::cout << "\n--- Test 9: NoiseFilter ---\n";

    // 9a. Standard octave centers include 1000 Hz
    {
        std::vector<Real> centers;
        NoiseFilter::generate_octave_centers(centers);
        CHECK(centers.size() == 10, "Filter: 10 standard octave bands");
        // Check 1000 Hz is in the list
        bool found_1k = false;
        for (auto f : centers) {
            if (std::abs(f - 1000.0) < 1.0) found_1k = true;
        }
        CHECK(found_1k, "Filter: 1 kHz in octave centers");
    }

    // 9b. 1/3 octave centers
    {
        std::vector<Real> centers;
        NoiseFilter::generate_third_octave_centers(20.0, 20000.0, centers);
        CHECK(centers.size() > 20, "Filter: >20 third-octave bands in 20-20k");
        // Should include something near 1000 Hz
        bool near_1k = false;
        for (auto f : centers) {
            if (std::abs(f - 1000.0) < 10.0) near_1k = true;
        }
        CHECK(near_1k, "Filter: ~1 kHz in 1/3 octave centers");
    }

    // 9c. Band edges for octave (N=1): upper/lower ratio = 2
    {
        Real f_lo, f_hi;
        NoiseFilter::compute_band_edges(1000.0, 1, f_lo, f_hi);
        Real ratio = f_hi / f_lo;
        CHECK_NEAR(ratio, 2.0, 1e-10, "Filter: octave band ratio = 2");
    }

    // 9d. Band edges for 1/3 octave (N=3): upper/lower ratio = 2^(1/3)
    {
        Real f_lo, f_hi;
        NoiseFilter::compute_band_edges(1000.0, 3, f_lo, f_hi);
        Real ratio = f_hi / f_lo;
        Real expected = std::pow(2.0, 1.0 / 3.0);
        CHECK_NEAR(ratio, expected, 1e-10, "Filter: 1/3 octave ratio = 2^(1/3)");
    }

    // 9e. A-weighting at 1 kHz should be ~0 dB
    {
        Real Aw_1k = NoiseFilter::a_weighting_correction(1000.0);
        CHECK_NEAR(Aw_1k, 0.0, 0.2, "Filter: A-weight at 1 kHz ~0 dB");
    }

    // 9f. A-weighting at low frequencies is negative (attenuated)
    {
        Real Aw_100 = NoiseFilter::a_weighting_correction(100.0);
        CHECK(Aw_100 < -15.0, "Filter: A-weight at 100 Hz < -15 dB");
    }

    // 9g. A-weighting at very low frequency is strongly negative
    {
        Real Aw_31 = NoiseFilter::a_weighting_correction(31.5);
        CHECK(Aw_31 < -35.0, "Filter: A-weight at 31.5 Hz < -35 dB");
    }
}


// ============================================================================
// 10. AcousticBEM Tests
// ============================================================================

void test_10_acoustic_bem() {
    std::cout << "\n--- Test 10: AcousticBEM ---\n";

    Real rho = 1.225;
    Real c_sound = 343.0;
    AcousticBEM bem(rho, c_sound);

    // 10a. Green's function value at known distance
    // G(r=1, k=1) = exp(-i) / (4*pi) ~ (cos(1) - i*sin(1)) / (4*pi)
    {
        auto G = bem.greens_function(1.0, 1.0);
        Real expected_real = std::cos(1.0) / (4.0 * acoustic_constants::PI);
        Real expected_imag = -std::sin(1.0) / (4.0 * acoustic_constants::PI);
        CHECK_NEAR(G.real(), expected_real, 1e-10, "BEM: G real part");
        CHECK_NEAR(G.imag(), expected_imag, 1e-10, "BEM: G imag part");
    }

    // 10b. Green's function magnitude decays as 1/(4*pi*r)
    {
        auto G1 = bem.greens_function(1.0, 0.0);  // k=0: no oscillation
        auto G2 = bem.greens_function(2.0, 0.0);
        // |G1|/|G2| = 2
        Real ratio = std::abs(G1) / std::abs(G2);
        CHECK_NEAR(ratio, 2.0, 1e-10, "BEM: G 1/r decay");
    }

    // 10c. Green's function at r=0 is 0 (regularized)
    {
        auto G0 = bem.greens_function(0.0, 1.0);
        CHECK_NEAR(std::abs(G0), 0.0, 1e-15, "BEM: G(r=0)=0 regularized");
    }

    // 10d. BEM assembly: diagonal of H is 0.5 (smooth boundary)
    {
        BEMPanel panels[2];
        panels[0] = {{0,0,0}, {0,0,1}, 1.0};
        panels[1] = {{2,0,0}, {0,0,1}, 1.0};

        int n = 2;
        std::vector<AcousticBEM::Complex> H(4), G_mat(4);
        bem.assemble_bem(panels, n, 100.0, H.data(), G_mat.data());

        CHECK_NEAR(H[0].real(), 0.5, 1e-10, "BEM: H diagonal = 0.5");
        CHECK_NEAR(H[3].real(), 0.5, 1e-10, "BEM: H diagonal[1,1] = 0.5");
    }

    // 10e. BEM G matrix: off-diagonal uses Green's function
    {
        BEMPanel panels[2];
        panels[0] = {{0,0,0}, {0,0,1}, 1.0};
        panels[1] = {{5,0,0}, {0,0,1}, 1.0};

        int n = 2;
        std::vector<AcousticBEM::Complex> H(4), G_mat(4);
        bem.assemble_bem(panels, n, 100.0, H.data(), G_mat.data());

        // G[0,1] = G(r=5, k) * A where k = 2*pi*100/343
        Real k = 2.0 * acoustic_constants::PI * 100.0 / c_sound;
        auto G_expected = bem.greens_function(5.0, k) * 1.0;
        CHECK_NEAR(std::abs(G_mat[1] - G_expected), 0.0, 1e-10,
                   "BEM: G off-diagonal matches Green's function");
    }

    // 10f. BEM solve: uniform v_n on a small system gives non-zero pressure
    {
        BEMPanel panels[3];
        panels[0] = {{0,0,0}, {0,0,1}, 0.5};
        panels[1] = {{1,0,0}, {0,0,1}, 0.5};
        panels[2] = {{2,0,0}, {0,0,1}, 0.5};

        Real v_n[3] = {1.0, 1.0, 1.0};
        std::vector<AcousticBEM::Complex> pressure(3);
        bem.solve_bem(panels, 3, 500.0, v_n, pressure.data());

        // All pressures should be non-zero (driven by uniform velocity)
        bool all_nonzero = true;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(pressure[i]) < 1e-20) all_nonzero = false;
        }
        CHECK(all_nonzero, "BEM: solve gives non-zero pressure");
    }

    // 10g. Wavenumber
    {
        Real k = bem.wavenumber(1000.0);
        Real expected = 2.0 * acoustic_constants::PI * 1000.0 / c_sound;
        CHECK_NEAR(k, expected, 1e-10, "BEM: wavenumber at 1 kHz");
    }
}


// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 37: Acoustic Analysis Tests ===\n";

    test_7_noise_computation();
    test_8_acoustic_pressure();
    test_9_noise_filter();
    test_10_acoustic_bem();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
