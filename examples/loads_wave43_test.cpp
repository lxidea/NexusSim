/**
 * @file loads_wave43_test.cpp
 * @brief Wave 43: Advanced load types test suite (~40 tests)
 *
 * Tests for 5 load types:
 *  1. CentrifugalLoad       - body force from rotation
 *  2. CylindricalPressure   - radial pressure on pipe surfaces
 *  3. FluidPressure         - hydrostatic depth-dependent pressure
 *  4. LaserLoad             - moving Gaussian heat source
 *  5. BoltPreload           - assembly preload force
 *
 * Each load type is tested for:
 *  - Analytical/reference values
 *  - Zero-magnitude / degenerate edge cases
 *  - Directionality
 *  - Time-varying (ramp/lock) behavior
 *  - LoadWave43Manager integration
 */

#include <nexussim/fem/loads_wave43.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace nxs;
using namespace nxs::fem;

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
// Test helpers
// ============================================================================

static void zero_forces(Real* f, std::size_t n) {
    for (std::size_t i = 0; i < 3*n; ++i) f[i] = 0.0;
}

static void zero_heat(Real* h, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) h[i] = 0.0;
}

// ============================================================================
// 1. CentrifugalLoad Tests
// ============================================================================

static void test_centrifugal() {
    std::cout << "\n--- CentrifugalLoad ---\n";

    // Node at (1, 0, 0), axis = Z through origin, omega = 10 rad/s
    // r_vec = (1, 0, 0), r = 1 m
    // F = m * omega^2 * r_vec = 2.0 * 100 * (1, 0, 0) = (200, 0, 0)
    {
        CentrifugalLoad cl;
        cl.omega = 10.0;
        cl.axis_point[0] = cl.axis_point[1] = cl.axis_point[2] = 0.0;
        cl.axis_direction[0] = 0.0; cl.axis_direction[1] = 0.0; cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {1.0, 0.0, 0.0};
        Real mass[1] = {2.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(1.0, 1, pos, mass, forces);

        CHECK_NEAR(forces[0], 200.0, 1.0e-9, "CentrifugalLoad: Fx = m*omega^2*r = 200");
        CHECK_NEAR(forces[1], 0.0,   1.0e-9, "CentrifugalLoad: Fy = 0 (on X axis)");
        CHECK_NEAR(forces[2], 0.0,   1.0e-9, "CentrifugalLoad: Fz = 0 (no axial component)");
    }

    // Node at (0, 0, 5) — on the Z axis — force must be zero
    {
        CentrifugalLoad cl;
        cl.omega = 100.0;
        cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {0.0, 0.0, 5.0};
        Real mass[1] = {1.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(1.0, 1, pos, mass, forces);

        CHECK_NEAR(forces[0], 0.0, 1.0e-9, "CentrifugalLoad: on-axis Fx = 0");
        CHECK_NEAR(forces[1], 0.0, 1.0e-9, "CentrifugalLoad: on-axis Fy = 0");
        CHECK_NEAR(forces[2], 0.0, 1.0e-9, "CentrifugalLoad: on-axis Fz = 0");
    }

    // Axis offset: axis_point = (2, 0, 0), axis = Z, node at (3, 0, 0)
    // r_vec = (3-2, 0, 0) - (0*0, 0*0, 0) = (1, 0, 0), r = 1
    // F = 1 * 5^2 * (1, 0, 0) = (25, 0, 0)
    {
        CentrifugalLoad cl;
        cl.omega = 5.0;
        cl.axis_point[0] = 2.0; cl.axis_point[1] = 0.0; cl.axis_point[2] = 0.0;
        cl.axis_direction[0] = 0.0; cl.axis_direction[1] = 0.0; cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {3.0, 0.0, 0.0};
        Real mass[1] = {1.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(1.0, 1, pos, mass, forces);

        CHECK_NEAR(forces[0], 25.0, 1.0e-9, "CentrifugalLoad: offset axis Fx = 25");
        CHECK_NEAR(forces[1], 0.0,  1.0e-9, "CentrifugalLoad: offset axis Fy = 0");
    }

    // Curve scale = 0.5 halves the force
    {
        CentrifugalLoad cl;
        cl.omega = 10.0;
        cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {1.0, 0.0, 0.0};
        Real mass[1] = {2.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(0.5, 1, pos, mass, forces);

        // F = m * (omega^2 * scale) * r = 2 * (100 * 0.5) * 1 = 100
        CHECK_NEAR(forces[0], 100.0, 1.0e-9, "CentrifugalLoad: curve scale 0.5 -> Fx=100");
    }

    // omega = 0 -> zero force
    {
        CentrifugalLoad cl;
        cl.omega = 0.0;
        cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {3.0, 4.0, 0.0};
        Real mass[1] = {5.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(1.0, 1, pos, mass, forces);

        CHECK_NEAR(forces[0], 0.0, 1.0e-9, "CentrifugalLoad: omega=0 -> Fx=0");
        CHECK_NEAR(forces[1], 0.0, 1.0e-9, "CentrifugalLoad: omega=0 -> Fy=0");
    }

    // Multiple nodes: force is proportional to radial distance
    {
        CentrifugalLoad cl;
        cl.omega = 1.0;
        cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        // 2 nodes: at (1,0,0) and (2,0,0)
        Real pos[6] = {1.0, 0.0, 0.0,   2.0, 0.0, 0.0};
        Real mass[2] = {1.0, 1.0};
        Real forces[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        cl.apply(1.0, 2, pos, mass, forces);

        // Node 0: F = 1*1*1 = 1, Node 1: F = 1*1*2 = 2
        CHECK_NEAR(forces[0], 1.0, 1.0e-9, "CentrifugalLoad: 2-node r=1 -> Fx=1");
        CHECK_NEAR(forces[3], 2.0, 1.0e-9, "CentrifugalLoad: 2-node r=2 -> Fx=2");
    }

    // Radial vector: node at (3, 4, 5), Z axis -> r_vec = (3,4,0), r = 5
    {
        CentrifugalLoad cl;
        cl.omega = 2.0;
        cl.axis_direction[2] = 1.0;
        cl.normalize_axis();

        Real pos[3] = {3.0, 4.0, 5.0};
        Real mass[1] = {1.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cl.apply(1.0, 1, pos, mass, forces);

        // F = m*omega^2*(3,4,0) = 4*(3,4,0) = (12, 16, 0)
        CHECK_NEAR(forces[0], 12.0, 1.0e-9, "CentrifugalLoad: 3D node Fx=12");
        CHECK_NEAR(forces[1], 16.0, 1.0e-9, "CentrifugalLoad: 3D node Fy=16");
        CHECK_NEAR(forces[2],  0.0, 1.0e-9, "CentrifugalLoad: 3D node Fz=0 (no axial)");
    }
}

// ============================================================================
// 2. CylindricalPressure Tests
// ============================================================================

static void test_cylindrical_pressure() {
    std::cout << "\n--- CylindricalPressure ---\n";

    // Node at (1, 0, 0), Z-axis cylinder, p=100, area=1
    // r_hat = (1,0,0), F = 100*(1,0,0) = (100,0,0)
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 100.0;
        cp.area_per_node = 1.0;
        cp.node_set = {0};

        Real pos[3] = {1.0, 0.0, 0.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cp.apply(1.0, pos, forces);

        CHECK_NEAR(forces[0], 100.0, 1.0e-9, "CylPressure: Fx=100 for node at (1,0,0)");
        CHECK_NEAR(forces[1], 0.0,   1.0e-9, "CylPressure: Fy=0");
        CHECK_NEAR(forces[2], 0.0,   1.0e-9, "CylPressure: Fz=0");
    }

    // Node at (0, 2, 0): r_hat = (0,1,0)
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 50.0;
        cp.area_per_node = 2.0;
        cp.node_set = {0};

        Real pos[3] = {0.0, 2.0, 0.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cp.apply(1.0, pos, forces);

        // F = 50 * 2 * (0,1,0) = (0, 100, 0)
        CHECK_NEAR(forces[0], 0.0,   1.0e-9, "CylPressure: diagonal Fx=0");
        CHECK_NEAR(forces[1], 100.0, 1.0e-9, "CylPressure: diagonal Fy=100");
    }

    // On-axis node: no force (degenerate)
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 1000.0;
        cp.area_per_node = 1.0;
        cp.node_set = {0};

        Real pos[3] = {0.0, 0.0, 3.0};  // On Z-axis
        Real forces[3] = {0.0, 0.0, 0.0};

        cp.apply(1.0, pos, forces);

        CHECK_NEAR(forces[0], 0.0, 1.0e-9, "CylPressure: on-axis -> Fx=0");
        CHECK_NEAR(forces[1], 0.0, 1.0e-9, "CylPressure: on-axis -> Fy=0");
    }

    // Curve scale = 0 -> zero force
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 500.0;
        cp.area_per_node = 1.0;
        cp.node_set = {0};

        Real pos[3] = {1.0, 0.0, 0.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cp.apply(0.0, pos, forces);

        CHECK_NEAR(forces[0], 0.0, 1.0e-9, "CylPressure: scale=0 -> zero force");
    }

    // 2 nodes at 45 degrees: equal radial distance, forces should be symmetric
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 1.0;
        cp.area_per_node = 1.0;
        cp.node_set = {0, 1};

        Real pos[6] = {1.0, 0.0, 0.0,   0.0, 1.0, 0.0};
        Real forces[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        cp.apply(1.0, pos, forces);

        // Both should have |F| = 1
        Real mag0 = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
        Real mag1 = std::sqrt(forces[3]*forces[3] + forces[4]*forces[4] + forces[5]*forces[5]);
        CHECK_NEAR(mag0, 1.0, 1.0e-9, "CylPressure: 2-node symmetric |F0|=1");
        CHECK_NEAR(mag1, 1.0, 1.0e-9, "CylPressure: 2-node symmetric |F1|=1");
    }

    // pressure = 0 -> zero force regardless
    {
        CylindricalPressure cp;
        cp.axis_direction[2] = 1.0;
        cp.normalize_axis();
        cp.pressure = 0.0;
        cp.area_per_node = 1.0;
        cp.node_set = {0};

        Real pos[3] = {2.0, 3.0, 1.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        cp.apply(1.0, pos, forces);

        Real mag = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
        CHECK_NEAR(mag, 0.0, 1.0e-9, "CylPressure: pressure=0 -> zero force");
    }
}

// ============================================================================
// 3. FluidPressure Tests
// ============================================================================

static void test_fluid_pressure() {
    std::cout << "\n--- FluidPressure ---\n";

    // Node at depth 1 m below Z=0 free surface, rho=1000, g=9.81
    // P = 1000 * 9.81 * 1 = 9810 Pa
    // F = P * area * (-Z) = 9810 * 1 * (0,0,-1) -> forces[2] = -9810
    {
        FluidPressure fp;
        fp.fluid_density = 1000.0;
        fp.gravity = 9.81;
        fp.free_surface_height = 0.0;
        fp.free_surface_normal[0] = 0.0;
        fp.free_surface_normal[1] = 0.0;
        fp.free_surface_normal[2] = 1.0;
        fp.normalize_normal();
        fp.area_per_node = 1.0;
        fp.node_set = {0};

        Real pos[3] = {0.0, 0.0, -1.0};  // 1 m below Z=0
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(1.0, pos, forces);

        CHECK_NEAR(forces[0], 0.0,    1.0e-6, "FluidPressure: Fx=0 (vertical pressure)");
        CHECK_NEAR(forces[1], 0.0,    1.0e-6, "FluidPressure: Fy=0 (vertical pressure)");
        CHECK_NEAR(forces[2], -9810.0, 1.0e-4, "FluidPressure: Fz=-rho*g*d*A=-9810");
    }

    // Node above free surface: no force
    {
        FluidPressure fp;
        fp.fluid_density = 1000.0;
        fp.gravity = 9.81;
        fp.free_surface_height = 0.0;
        fp.free_surface_normal[2] = 1.0;
        fp.normalize_normal();
        fp.area_per_node = 1.0;
        fp.node_set = {0};

        Real pos[3] = {0.0, 0.0, 1.0};  // 1 m above Z=0
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(1.0, pos, forces);

        CHECK_NEAR(forces[2], 0.0, 1.0e-12, "FluidPressure: above surface -> Fz=0");
    }

    // At the free surface exactly: depth=0, no force
    {
        FluidPressure fp;
        fp.fluid_density = 1000.0;
        fp.gravity = 9.81;
        fp.free_surface_height = 0.0;
        fp.free_surface_normal[2] = 1.0;
        fp.normalize_normal();
        fp.area_per_node = 1.0;
        fp.node_set = {0};

        Real pos[3] = {5.0, 3.0, 0.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(1.0, pos, forces);

        CHECK_NEAR(forces[2], 0.0, 1.0e-12, "FluidPressure: at surface depth=0 -> Fz=0");
    }

    // Non-zero free_surface_height: surface at Z=2, node at Z=1 -> depth=1
    {
        FluidPressure fp;
        fp.fluid_density = 1000.0;
        fp.gravity = 10.0;
        fp.free_surface_height = 2.0;
        fp.free_surface_normal[2] = 1.0;
        fp.normalize_normal();
        fp.area_per_node = 1.0;
        fp.node_set = {0};

        Real pos[3] = {0.0, 0.0, 1.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(1.0, pos, forces);

        // depth = 2.0 - 1.0 = 1.0, P = 1000*10*1 = 10000
        CHECK_NEAR(forces[2], -10000.0, 1.0e-4, "FluidPressure: elevated surface Z=2 -> Fz=-10000");
    }

    // Curve scale = 0.5
    {
        FluidPressure fp;
        fp.fluid_density = 1000.0;
        fp.gravity = 9.81;
        fp.free_surface_height = 0.0;
        fp.free_surface_normal[2] = 1.0;
        fp.normalize_normal();
        fp.area_per_node = 1.0;
        fp.node_set = {0};

        Real pos[3] = {0.0, 0.0, -2.0};  // depth = 2 m
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(0.5, pos, forces);

        // P = 1000*9.81*2*0.5 = 9810
        CHECK_NEAR(forces[2], -9810.0, 1.0e-4, "FluidPressure: curve_scale=0.5 depth=2 -> Fz=-9810");
    }

    // Horizontal free surface normal (Y-up): node at Y=-3 -> depth=3
    {
        FluidPressure fp;
        fp.fluid_density = 800.0;
        fp.gravity = 9.81;
        fp.free_surface_height = 0.0;
        fp.free_surface_normal[0] = 0.0;
        fp.free_surface_normal[1] = 1.0;
        fp.free_surface_normal[2] = 0.0;
        fp.normalize_normal();
        fp.area_per_node = 2.0;
        fp.node_set = {0};

        Real pos[3] = {1.0, -3.0, 2.0};
        Real forces[3] = {0.0, 0.0, 0.0};

        fp.apply(1.0, pos, forces);

        // depth = 0 - (-3) = 3, P = 800*9.81*3 = 23544, F = 23544*2 = 47088 in -Y
        double expected = -800.0 * 9.81 * 3.0 * 2.0;
        CHECK_NEAR(forces[1], expected, 1.0e-3, "FluidPressure: Y-up normal depth=3 correct");
        CHECK_NEAR(forces[0], 0.0, 1.0e-9, "FluidPressure: Y-up Fx=0");
    }
}

// ============================================================================
// 4. LaserLoad Tests
// ============================================================================

static void test_laser_load() {
    std::cout << "\n--- LaserLoad ---\n";

    // Peak intensity at beam center: Q_peak = P/(2*pi*sigma^2)
    // Node exactly at beam center, area=1, absorptivity=1, t=0 (beam at position[0,0,0])
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI;  // So peak = 1/(sigma^2) = 1 for sigma=1
        ll.beam_radius = 1.0;   // sigma=1
        ll.position[0] = ll.position[1] = ll.position[2] = 0.0;
        ll.direction[0] = 1.0; ll.direction[1] = 0.0; ll.direction[2] = 0.0;
        ll.speed = 0.0;
        ll.absorption_coeff = 1.0;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {0.0, 0.0, 0.0};
        Real heat[1] = {0.0};

        ll.apply(0.0, 1.0, pos, heat, nullptr);

        // Q = P/(2*pi*sigma^2) * exp(0) * area = (2*pi)/(2*pi*1) * 1 * 1 = 1.0
        CHECK_NEAR(heat[0], 1.0, 1.0e-9, "LaserLoad: peak at center Q=1");
    }

    // At r = sigma: Q = peak * exp(-0.5) ≈ 0.6065 * peak
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI;
        ll.beam_radius = 1.0;
        ll.position[0] = ll.position[1] = ll.position[2] = 0.0;
        ll.speed = 0.0;
        ll.absorption_coeff = 1.0;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {1.0, 0.0, 0.0};  // r = 1 = sigma
        Real heat[1] = {0.0};

        ll.apply(0.0, 1.0, pos, heat, nullptr);

        double expected = std::exp(-0.5);
        CHECK_NEAR(heat[0], expected, 1.0e-9, "LaserLoad: r=sigma Q=peak*exp(-0.5)");
    }

    // Moving beam: at t=1, speed=1, direction=X -> beam center at (1,0,0)
    // Node at (1,0,0) should get peak intensity
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI;
        ll.beam_radius = 1.0;
        ll.position[0] = ll.position[1] = ll.position[2] = 0.0;
        ll.direction[0] = 1.0; ll.direction[1] = 0.0; ll.direction[2] = 0.0;
        ll.speed = 1.0;
        ll.absorption_coeff = 1.0;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {1.0, 0.0, 0.0};
        Real heat[1] = {0.0};

        ll.apply(1.0, 1.0, pos, heat, nullptr);

        // Beam center at (1,0,0), node at (1,0,0) -> r=0 -> peak=1
        CHECK_NEAR(heat[0], 1.0, 1.0e-9, "LaserLoad: moving beam t=1 peak at (1,0,0)");
    }

    // power=0 -> zero flux
    {
        LaserLoad ll;
        ll.power = 0.0;
        ll.beam_radius = 1.0;
        ll.node_set = {0};

        Real pos[3] = {0.5, 0.0, 0.0};
        Real heat[1] = {0.0};

        ll.apply(0.0, 1.0, pos, heat, nullptr);

        CHECK_NEAR(heat[0], 0.0, 1.0e-12, "LaserLoad: power=0 -> zero flux");
    }

    // absorption_coeff = 0.5 halves the heat flux
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI;
        ll.beam_radius = 1.0;
        ll.speed = 0.0;
        ll.absorption_coeff = 0.5;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {0.0, 0.0, 0.0};
        Real heat[1] = {0.0};

        ll.apply(0.0, 1.0, pos, heat, nullptr);

        CHECK_NEAR(heat[0], 0.5, 1.0e-9, "LaserLoad: absorptivity=0.5 -> Q=0.5");
    }

    // curve_scale = 2.0 doubles the heat flux
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI;
        ll.beam_radius = 1.0;
        ll.speed = 0.0;
        ll.absorption_coeff = 1.0;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {0.0, 0.0, 0.0};
        Real heat[1] = {0.0};

        ll.apply(0.0, 2.0, pos, heat, nullptr);

        CHECK_NEAR(heat[0], 2.0, 1.0e-9, "LaserLoad: curve_scale=2 -> Q=2");
    }

    // Gaussian decay: node far from beam (r >> sigma) -> near zero
    {
        LaserLoad ll;
        ll.power = 2.0 * M_PI * 1.0e6;  // Large power
        ll.beam_radius = 0.001;           // Tight beam
        ll.speed = 0.0;
        ll.absorption_coeff = 1.0;
        ll.area_per_node = 1.0;
        ll.node_set = {0};

        Real pos[3] = {1.0, 0.0, 0.0};  // r = 1 >> sigma = 0.001
        Real heat[1] = {0.0};

        ll.apply(0.0, 1.0, pos, heat, nullptr);

        CHECK(heat[0] < 1.0e-100, "LaserLoad: node far from beam -> near-zero flux");
    }
}

// ============================================================================
// 5. BoltPreload Tests
// ============================================================================

static void test_bolt_preload() {
    std::cout << "\n--- BoltPreload ---\n";

    // At t=0 (start of ramp), ramp_factor = 0 -> zero force
    {
        BoltPreload bp;
        bp.preload_force = 10000.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};

        bp.apply(0.0, 1.0, forces);

        CHECK_NEAR(forces[2], 0.0, 1.0e-9, "BoltPreload: t=0 -> F=0 (start of ramp)");
        CHECK_NEAR(forces[5], 0.0, 1.0e-9, "BoltPreload: t=0 nut node -> F=0");
    }

    // At t=ramp_time/2, ramp_factor = 0.5 -> half force
    {
        BoltPreload bp;
        bp.preload_force = 10000.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};

        bp.apply(0.5, 1.0, forces);

        // head: +5000 in Z, nut: -5000 in Z
        CHECK_NEAR(forces[2],  5000.0, 1.0e-6, "BoltPreload: t=0.5 head Fz=5000");
        CHECK_NEAR(forces[5], -5000.0, 1.0e-6, "BoltPreload: t=0.5 nut  Fz=-5000");
    }

    // At t >= ramp_time: ramp_factor = 1.0 -> full force (not yet locked)
    {
        BoltPreload bp;
        bp.preload_force = 10000.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};

        bp.apply(1.0, 1.0, forces);

        CHECK_NEAR(forces[2],  10000.0, 1.0e-6, "BoltPreload: t=ramp_time full force head");
        CHECK_NEAR(forces[5], -10000.0, 1.0e-6, "BoltPreload: t=ramp_time full force nut");
    }

    // After lock: zero incremental force
    {
        BoltPreload bp;
        bp.preload_force = 10000.0;
        bp.ramp_time = 0.001;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};

        bp.update_lock_state(0.002);  // Past ramp time -> locked
        bp.apply(0.002, 1.0, forces);

        CHECK(bp.locked, "BoltPreload: state locked after ramp_time");
        CHECK_NEAR(forces[2], 0.0, 1.0e-9, "BoltPreload: locked -> zero incremental force");
    }

    // Multiple head nodes: force split equally
    {
        BoltPreload bp;
        bp.preload_force = 9000.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0, 1, 2};  // 3 nodes
        // nut_nodes empty

        Real forces[9] = {0,0,0, 0,0,0, 0,0,0};

        bp.apply(1.0, 1.0, forces);

        // Each head node: 9000/3 = 3000 in Z
        CHECK_NEAR(forces[2], 3000.0, 1.0e-6, "BoltPreload: 3 head nodes Fz[0]=3000");
        CHECK_NEAR(forces[5], 3000.0, 1.0e-6, "BoltPreload: 3 head nodes Fz[1]=3000");
        CHECK_NEAR(forces[8], 3000.0, 1.0e-6, "BoltPreload: 3 head nodes Fz[2]=3000");
    }

    // Non-Z bolt axis: bolt_axis = (1,0,0)
    {
        BoltPreload bp;
        bp.preload_force = 5000.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[0] = 1.0; bp.bolt_axis[1] = 0.0; bp.bolt_axis[2] = 0.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};

        bp.apply(1.0, 1.0, forces);

        // head: +5000 in X
        CHECK_NEAR(forces[0],  5000.0, 1.0e-6, "BoltPreload: X-axis head Fx=5000");
        CHECK_NEAR(forces[3], -5000.0, 1.0e-6, "BoltPreload: X-axis nut  Fx=-5000");
        CHECK_NEAR(forces[2], 0.0, 1.0e-9, "BoltPreload: X-axis Fz=0");
    }

    // preload_force = 0 -> zero force
    {
        BoltPreload bp;
        bp.preload_force = 0.0;
        bp.ramp_time = 1.0;
        bp.bolt_axis[2] = 1.0;
        bp.normalize_axis();
        bp.head_nodes = {0};
        bp.nut_nodes  = {1};

        Real forces[6] = {0,0,0, 0,0,0};
        bp.apply(0.8, 1.0, forces);

        CHECK_NEAR(forces[2], 0.0, 1.0e-12, "BoltPreload: preload=0 -> zero force");
    }
}

// ============================================================================
// 6. LoadWave43Manager Integration Tests
// ============================================================================

static void test_manager() {
    std::cout << "\n--- LoadWave43Manager ---\n";

    // Add all 5 types and verify counts
    {
        LoadWave43Manager mgr;

        mgr.add_load(LoadWave43Type::CentrifugalLoad,     1).name = "rotor_cf";
        mgr.add_load(LoadWave43Type::CylindricalPressure, 2).name = "pipe_press";
        mgr.add_load(LoadWave43Type::FluidPressure,       3).name = "hydro";
        mgr.add_load(LoadWave43Type::LaserLoad,           4).name = "weld_laser";
        mgr.add_load(LoadWave43Type::BoltPreload,         5).name = "bolt1";

        CHECK(mgr.num_loads() == 5, "Manager: 5 loads registered");
        CHECK(mgr.load(0).type == LoadWave43Type::CentrifugalLoad,
              "Manager: load[0] is CentrifugalLoad");
        CHECK(mgr.load(4).type == LoadWave43Type::BoltPreload,
              "Manager: load[4] is BoltPreload");
    }

    // apply_all: centrifugal + fluid pressure on separate nodes
    {
        LoadWave43Manager mgr;

        // CentrifugalLoad: Z axis, omega=10, node 0 at (1,0,0), mass=2
        auto& cl = mgr.add_load(LoadWave43Type::CentrifugalLoad, 1);
        cl.centrifugal.omega = 10.0;
        cl.centrifugal.axis_direction[2] = 1.0;
        cl.centrifugal.normalize_axis();

        // FluidPressure: node 1 at (0,0,-1), depth=1, rho=1000, g=9.81, area=1
        auto& fl = mgr.add_load(LoadWave43Type::FluidPressure, 2);
        fl.fluid_pressure.fluid_density = 1000.0;
        fl.fluid_pressure.gravity = 9.81;
        fl.fluid_pressure.free_surface_height = 0.0;
        fl.fluid_pressure.free_surface_normal[2] = 1.0;
        fl.fluid_pressure.normalize_normal();
        fl.fluid_pressure.area_per_node = 1.0;
        fl.fluid_pressure.node_set = {1};

        Real pos[6]    = {1.0, 0.0, 0.0,   0.0, 0.0, -1.0};
        Real masses[2] = {2.0, 1.0};
        Real forces[6] = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};

        mgr.apply_all(0.0, 2, pos, masses, forces, nullptr);

        // Node 0 centrifugal: Fx = 2*100*1 = 200
        CHECK_NEAR(forces[0], 200.0, 1.0e-9, "Manager: centrifugal node0 Fx=200");
        // Node 1 fluid: Fz = -9810
        CHECK_NEAR(forces[5], -9810.0, 1.0e-4, "Manager: fluid node1 Fz=-9810");
    }

    // print_summary does not crash
    {
        LoadWave43Manager mgr;
        mgr.add_load(LoadWave43Type::LaserLoad, 10).name = "test_laser";
        mgr.print_summary();
        tests_passed++;  // Reached without crash
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== Wave 43: Advanced Load Types Test Suite ===\n";

    test_centrifugal();
    test_cylindrical_pressure();
    test_fluid_pressure();
    test_laser_load();
    test_bolt_preload();
    test_manager();

    std::cout << "\n=== Results: "
              << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
