/**
 * @file composite_layup_test.cpp
 * @brief Comprehensive test for Wave 7: Section properties and composite layup
 */

#include <nexussim/physics/section.hpp>
#include <nexussim/physics/composite_layup.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "[PASS] " << msg << "\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << "\n"; \
        tests_failed++; \
    } \
} while(0)

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(b));
}

static bool near_rel(Real a, Real b, Real rel_tol = 0.01) {
    if (std::fabs(b) < 1.0e-30) return std::fabs(a) < 1.0e-20;
    return std::fabs(a - b) / std::fabs(b) < rel_tol;
}

// ==========================================================================
// Test 1: Rectangular Beam Section
// ==========================================================================
void test_beam_rectangular() {
    std::cout << "\n=== Test 1: Rectangular Beam Section ===\n";

    SectionProperties sec;
    sec.type = SectionType::BeamRectangular;
    sec.width = 0.02;   // 20 mm
    sec.height = 0.05;  // 50 mm
    sec.compute();

    CHECK(near(sec.area, 0.001), "Area = 20mm × 50mm = 1000 mm²");
    CHECK(near(sec.Iyy, 0.02*0.05*0.05*0.05/12.0), "Iyy = bh³/12");
    CHECK(near(sec.Izz, 0.05*0.02*0.02*0.02/12.0), "Izz = hb³/12");
    CHECK(sec.J > 0.0, "Torsional constant > 0");
    CHECK(near(sec.ky, 5.0/6.0), "Shear factor ky = 5/6");
}

// ==========================================================================
// Test 2: Circular Beam Section
// ==========================================================================
void test_beam_circular() {
    std::cout << "\n=== Test 2: Circular Beam Section ===\n";

    SectionProperties sec;
    sec.type = SectionType::BeamCircular;
    sec.diameter = 0.01;  // 10 mm diameter
    sec.compute();

    Real r = 0.005;
    Real pi = constants::pi<Real>;
    CHECK(near(sec.area, pi * r * r), "Area = πr²");
    CHECK(near(sec.Iyy, pi * r * r * r * r / 4.0), "Iyy = πr⁴/4");
    CHECK(near(sec.Iyy, sec.Izz), "Iyy = Izz (symmetric)");
    CHECK(near(sec.J, pi * r * r * r * r / 2.0), "J = πr⁴/2 (polar)");
    CHECK(near(sec.ky, 6.0/7.0), "Shear factor = 6/7 for circle");
}

// ==========================================================================
// Test 3: Hollow Circular (Tube)
// ==========================================================================
void test_beam_hollow() {
    std::cout << "\n=== Test 3: Hollow Circular Section ===\n";

    SectionProperties sec;
    sec.type = SectionType::BeamHollowCircular;
    sec.diameter = 0.050;       // 50 mm outer diameter
    sec.wall_thickness = 0.005; // 5 mm wall
    sec.compute();

    Real ro = 0.025, ri = 0.020;
    Real pi = constants::pi<Real>;
    CHECK(near(sec.area, pi*(ro*ro - ri*ri)), "Area = π(ro²-ri²)");
    CHECK(sec.Iyy > 0.0, "Iyy > 0");
    CHECK(near(sec.Iyy, sec.Izz), "Iyy = Izz (axisymmetric)");

    // Hollow tube has more efficient Iyy/area ratio
    SectionProperties solid;
    solid.type = SectionType::BeamCircular;
    solid.diameter = 0.050;
    solid.compute();
    CHECK(sec.area < solid.area, "Hollow area < solid area");
    // Iyy should be close for thin-walled tube
    CHECK(sec.Iyy > 0.5 * solid.Iyy, "Hollow Iyy > 50% of solid Iyy");
}

// ==========================================================================
// Test 4: I-Beam Section
// ==========================================================================
void test_beam_ibeam() {
    std::cout << "\n=== Test 4: I-Beam Section ===\n";

    SectionProperties sec;
    sec.type = SectionType::BeamIBeam;
    sec.height = 0.200;           // 200 mm total height
    sec.flange_width = 0.100;     // 100 mm flange
    sec.flange_thickness = 0.010; // 10 mm flange thickness
    sec.web_thickness = 0.006;    // 6 mm web
    sec.compute();

    // Area = 2*flange + web
    Real hw = 0.200 - 2*0.010;
    Real expected_area = 2.0*0.100*0.010 + hw*0.006;
    CHECK(near(sec.area, expected_area), "Area = 2*flange_area + web_area");
    CHECK(sec.Iyy > sec.Izz, "Iyy > Izz (strong axis bending)");
    CHECK(sec.J > 0.0, "Torsional constant > 0");
}

// ==========================================================================
// Test 5: Box Section
// ==========================================================================
void test_beam_box() {
    std::cout << "\n=== Test 5: Box Section ===\n";

    SectionProperties sec;
    sec.type = SectionType::BeamBoxSection;
    sec.width = 0.060;
    sec.height = 0.100;
    sec.wall_thickness = 0.005;
    sec.compute();

    Real wi = 0.060 - 2*0.005;
    Real hi = 0.100 - 2*0.005;
    Real expected_area = 0.060*0.100 - wi*hi;
    CHECK(near(sec.area, expected_area), "Area = outer - inner");
    CHECK(sec.Iyy > 0.0, "Iyy > 0");
    CHECK(sec.J > 0.0, "J > 0 (Bredt's formula)");
    CHECK(sec.Iyy > sec.Izz, "Iyy > Izz (taller than wide)");
}

// ==========================================================================
// Test 6: Shell Gauss Integration
// ==========================================================================
void test_shell_integration() {
    std::cout << "\n=== Test 6: Shell Gauss Integration ===\n";

    SectionProperties sec;
    sec.type = SectionType::ShellUniform;
    sec.thickness = 0.002;

    // 2-point Gauss
    sec.num_ip_thickness = 2;
    sec.compute();
    CHECK(sec.integration.num_points == 2, "2-point integration");

    // Verify weights sum to 1
    Real sum_w = 0.0;
    for (int i = 0; i < sec.integration.num_points; ++i)
        sum_w += sec.integration.points[i].weight;
    CHECK(near(sum_w, 1.0), "Weights sum to 1.0");

    // Points should be symmetric about 0
    CHECK(near(sec.integration.points[0].z, -sec.integration.points[1].z, 1e-10),
          "Points symmetric about midplane");

    // 5-point Gauss
    sec.num_ip_thickness = 5;
    sec.compute();
    CHECK(sec.integration.num_points == 5, "5-point integration");
    sum_w = 0.0;
    for (int i = 0; i < 5; ++i)
        sum_w += sec.integration.points[i].weight;
    CHECK(near(sum_w, 1.0, 1e-10), "5-point weights sum to 1.0");
}

// ==========================================================================
// Test 7: Variable Thickness Shell
// ==========================================================================
void test_variable_thickness() {
    std::cout << "\n=== Test 7: Variable Thickness Shell ===\n";

    SectionProperties sec;
    sec.type = SectionType::ShellVariable;
    sec.thickness_nodes[0] = 0.001;
    sec.thickness_nodes[1] = 0.002;
    sec.thickness_nodes[2] = 0.003;
    sec.thickness_nodes[3] = 0.004;

    // Center: average of corners
    Real t_center = sec.interpolate_thickness(0.0, 0.0);
    CHECK(near(t_center, 0.0025), "Center thickness = average = 2.5mm");

    // Corner 0 (xi=-1, eta=-1)
    Real t0 = sec.interpolate_thickness(-1.0, -1.0);
    CHECK(near(t0, 0.001), "Corner 0: t = 1mm");

    // Corner 2 (xi=+1, eta=+1)
    Real t2 = sec.interpolate_thickness(1.0, 1.0);
    CHECK(near(t2, 0.003), "Corner 2: t = 3mm");
}

// ==========================================================================
// Test 8: Unidirectional Composite [0]4 - ABD Matrix
// ==========================================================================
void test_unidirectional_abd() {
    std::cout << "\n=== Test 8: Unidirectional [0]4 ABD ===\n";

    // Carbon fiber/epoxy ply properties
    Real E1 = 138.0e9;     // 138 GPa longitudinal
    Real E2 = 8.96e9;      // 8.96 GPa transverse
    Real G12 = 7.10e9;     // 7.10 GPa shear
    Real nu12 = 0.30;
    Real ply_t = 0.000125;  // 0.125 mm ply thickness

    auto lam = layup_presets::unidirectional(E1, E2, G12, nu12, ply_t, 4);
    lam.compute_abd();

    CHECK(lam.num_plies() == 4, "4 plies");
    CHECK(near(lam.total_thickness(), 4.0*ply_t), "Total thickness = 4 × 0.125mm");

    // For UD [0]n, A16 = A26 = 0 (no shear coupling)
    CHECK(near(lam.A()[2], 0.0, 1.0), "A16 ≈ 0 for UD [0]");
    CHECK(near(lam.A()[5], 0.0, 1.0), "A26 ≈ 0 for UD [0]");

    // B should be zero for symmetric layup (UD is symmetric)
    // Actually [0]4 is symmetric about midplane by construction
    CHECK(lam.is_symmetric(), "[0]4 is symmetric");

    // A11 should be large (fiber direction), A22 small
    CHECK(lam.A()[0] > lam.A()[4], "A11 > A22 (stiffer in fiber direction)");

    // Effective properties
    auto ep = lam.effective_properties();
    CHECK(near_rel(ep.Ex, E1, 0.01), "Ex ≈ E1 for UD [0]");
    CHECK(near_rel(ep.Ey, E2, 0.01), "Ey ≈ E2 for UD [0]");
    CHECK(near_rel(ep.Gxy, G12, 0.01), "Gxy ≈ G12 for UD [0]");
}

// ==========================================================================
// Test 9: Cross-Ply [0/90]s - Symmetry and Balance
// ==========================================================================
void test_cross_ply() {
    std::cout << "\n=== Test 9: Cross-Ply [0/90]s ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CHECK(lam.num_plies() == 4, "[0/90]s has 4 plies");
    CHECK(lam.is_symmetric(), "Cross-ply [0/90]s is symmetric");
    CHECK(lam.is_balanced(), "Cross-ply is balanced (A16=A26=0)");

    // For [0/90]s: Ex ≈ Ey (approximately, because of equal 0° and 90° plies)
    auto ep = lam.effective_properties();
    CHECK(near_rel(ep.Ex, ep.Ey, 0.01), "Ex ≈ Ey for balanced cross-ply");

    // B should be zero (symmetric)
    for (int i = 0; i < 9; ++i) {
        CHECK(near(lam.B()[i], 0.0, 1.0), "B entries ≈ 0 for symmetric layup");
    }

    // Ex should be between E1 and E2
    CHECK(ep.Ex > E2 && ep.Ex < E1, "E2 < Ex < E1 for cross-ply");
}

// ==========================================================================
// Test 10: Quasi-Isotropic [0/±45/90]s
// ==========================================================================
void test_quasi_isotropic() {
    std::cout << "\n=== Test 10: Quasi-Isotropic [0/±45/90]s ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CHECK(lam.num_plies() == 8, "[0/+45/-45/90]s has 8 plies");
    CHECK(lam.is_symmetric(), "QI layup is symmetric");

    // For quasi-isotropic: Ex ≈ Ey and nuxy reasonable
    auto ep = lam.effective_properties();
    CHECK(near_rel(ep.Ex, ep.Ey, 0.02), "Ex ≈ Ey for quasi-isotropic");
    CHECK(ep.nuxy > 0.0 && ep.nuxy < 0.5, "0 < nuxy < 0.5");

    // Gxy ≈ Ex / (2*(1+nu)) for isotropic behavior
    Real G_iso = ep.Ex / (2.0 * (1.0 + ep.nuxy));
    CHECK(near_rel(ep.Gxy, G_iso, 0.10), "Gxy ≈ Ex/(2(1+ν)) within 10%");
}

// ==========================================================================
// Test 11: Angle-Ply [±45]s
// ==========================================================================
void test_angle_ply() {
    std::cout << "\n=== Test 11: Angle-Ply [±45]s ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::angle_ply(E1, E2, G12, nu12, ply_t, 45.0, 1);
    lam.compute_abd();

    CHECK(lam.num_plies() == 4, "[±45]s has 4 plies");
    CHECK(lam.is_symmetric(), "Symmetric");
    CHECK(lam.is_balanced(), "Balanced (±45 cancel A16/A26)");

    auto ep = lam.effective_properties();
    // [±45] has high shear stiffness
    CHECK(ep.Gxy > G12, "Gxy > G12 for [±45]");
    // Ex should be between pure 0° and pure 90° values
    CHECK(ep.Ex > E2 && ep.Ex < E1, "E2 < Ex < E1 for [±45]");
}

// ==========================================================================
// Test 12: Ply Stress Recovery
// ==========================================================================
void test_ply_stress_recovery() {
    std::cout << "\n=== Test 12: Ply Stress Recovery ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    // Apply uniaxial extension: ε⁰ = [0.001, 0, 0], κ = [0, 0, 0]
    Real eps0[3] = {0.001, 0.0, 0.0};
    Real kappa[3] = {0.0, 0.0, 0.0};

    PlyState states[4];
    int n = lam.compute_ply_stresses(eps0, kappa, states);
    CHECK(n == 4, "4 ply states computed");

    // 0° plies should have high σ11 (fiber direction = loading direction)
    // 90° plies should have low σ11 (fiber perpendicular to loading)
    // For [0/90]s: plies 0,3 are 0°, plies 1,2 are 90°
    CHECK(states[0].stress_global[0] > 0.0, "0° ply: positive σxx");
    CHECK(states[1].stress_global[0] > 0.0, "90° ply: positive σxx (but smaller)");

    // In local coords, 0° ply: σ11 high (fiber loaded)
    CHECK(states[0].stress_local[0] > states[1].stress_local[0],
          "0° ply σ11 > 90° ply σ11 in local coords");

    // Verify through-thickness positions are distinct
    for (int i = 0; i < 3; ++i) {
        CHECK(states[i].z_position < states[i+1].z_position,
              "Through-thickness positions increasing");
    }
}

// ==========================================================================
// Test 13: Force/Moment Resultants
// ==========================================================================
void test_resultants() {
    std::cout << "\n=== Test 13: Force/Moment Resultants ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    // Pure extension: ε⁰ = [0.001, 0, 0], κ = 0
    Real eps0[3] = {0.001, 0.0, 0.0};
    Real kappa[3] = {0.0, 0.0, 0.0};
    Real N[3], M[3];

    lam.compute_resultants(eps0, kappa, N, M);

    CHECK(N[0] > 0.0, "Nxx > 0 for tensile extension");
    CHECK(near(N[2], 0.0, 1.0), "Nxy ≈ 0 for balanced laminate");

    // For symmetric laminate with pure extension, M should be 0
    for (int i = 0; i < 3; ++i) {
        CHECK(near(M[i], 0.0, 0.1), "M ≈ 0 for symmetric + pure extension");
    }

    // Nxx = A11 * ε⁰_xx
    CHECK(near(N[0], lam.A()[0] * 0.001, 1.0), "Nxx = A11 * εxx");

    // Pure bending: κ = [1.0, 0, 0]
    Real eps0_zero[3] = {0.0, 0.0, 0.0};
    Real kappa_bend[3] = {1.0, 0.0, 0.0};

    lam.compute_resultants(eps0_zero, kappa_bend, N, M);
    CHECK(M[0] > 0.0, "Mxx > 0 for positive curvature");
    CHECK(near(M[0], lam.D()[0] * 1.0, 1.0), "Mxx = D11 * κxx");
}

// ==========================================================================
// Test 14: Unsymmetric Laminate (Non-Zero B)
// ==========================================================================
void test_unsymmetric() {
    std::cout << "\n=== Test 14: Unsymmetric Laminate ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    // [0/90] (NOT symmetric - no mirror)
    CompositeLaminate lam;
    PlyDefinition p0(E1, E2, G12, nu12, ply_t, 0.0);
    PlyDefinition p90(E1, E2, G12, nu12, ply_t, 90.0);
    lam.add_ply(p0);
    lam.add_ply(p90);
    lam.compute_abd();

    CHECK(!lam.is_symmetric(), "[0/90] is NOT symmetric");

    // B matrix should be non-zero
    bool has_nonzero_B = false;
    for (int i = 0; i < 9; ++i) {
        if (std::fabs(lam.B()[i]) > 1.0) { has_nonzero_B = true; break; }
    }
    CHECK(has_nonzero_B, "B ≠ 0 for unsymmetric laminate");

    // Extension-bending coupling: pure extension → non-zero moments
    Real eps0[3] = {0.001, 0.0, 0.0};
    Real kappa[3] = {0.0, 0.0, 0.0};
    Real N[3], M[3];
    lam.compute_resultants(eps0, kappa, N, M);

    bool nonzero_M = false;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(M[i]) > 0.01) { nonzero_M = true; break; }
    }
    CHECK(nonzero_M, "Extension causes bending in unsymmetric laminate");
}

// ==========================================================================
// Test 15: Composite Section Integration Setup
// ==========================================================================
void test_composite_section() {
    std::cout << "\n=== Test 15: Composite Section Integration ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);

    SectionProperties sec;
    lam.setup_integration(sec, 1);

    CHECK(sec.type == SectionType::ShellComposite, "Section type is composite");
    CHECK(near(sec.thickness, lam.total_thickness()), "Section thickness matches laminate");
    CHECK(sec.integration.num_points == 8, "8 integration points (1 per ply)");

    // Each point should have a ply_id
    for (int i = 0; i < 8; ++i) {
        CHECK(sec.integration.points[i].ply_id == i,
              "Integration point ply_id correct");
    }

    // Weights should sum to 1.0
    Real sum_w = 0.0;
    for (int i = 0; i < 8; ++i)
        sum_w += sec.integration.points[i].weight;
    CHECK(near(sum_w, 1.0, 1e-10), "Integration weights sum to 1.0");
}

// ==========================================================================
// Test 16: Section Manager
// ==========================================================================
void test_section_manager() {
    std::cout << "\n=== Test 16: Section Manager ===\n";

    SectionManager mgr;

    auto& s1 = mgr.add_section(1);
    s1.name = "Shell_2mm";
    s1.type = SectionType::ShellUniform;
    s1.thickness = 0.002;
    s1.num_ip_thickness = 3;

    auto& s2 = mgr.add_section(2);
    s2.name = "Beam_rect";
    s2.type = SectionType::BeamRectangular;
    s2.width = 0.01;
    s2.height = 0.02;

    mgr.compute_all();

    CHECK(mgr.num_sections() == 2, "2 sections");

    auto* found = mgr.find(1);
    CHECK(found != nullptr, "Section 1 found");
    CHECK(found->integration.num_points == 3, "Section 1 has 3 IPs");

    auto* found2 = mgr.find(2);
    CHECK(found2 != nullptr, "Section 2 found");
    CHECK(near(found2->area, 0.0002), "Section 2 area = 200 mm²");

    auto* not_found = mgr.find(99);
    CHECK(not_found == nullptr, "Section 99 not found");
}

// ==========================================================================
// Test 17: ABD Matrix Structure Verification
// ==========================================================================
void test_abd_matrix_structure() {
    std::cout << "\n=== Test 17: ABD Matrix Structure ===\n";

    Real E1 = 138.0e9, E2 = 8.96e9, G12 = 7.10e9, nu12 = 0.30;
    Real ply_t = 0.000125;

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    Real abd[36];
    lam.get_abd_matrix(abd);

    // ABD should be symmetric: abd[i][j] = abd[j][i]
    for (int i = 0; i < 6; ++i) {
        for (int j = i+1; j < 6; ++j) {
            CHECK(near(abd[i*6+j], abd[j*6+i], 0.01),
                  "ABD symmetry: [" + std::to_string(i) + "][" + std::to_string(j) + "]");
        }
    }

    // Diagonal should be positive
    for (int i = 0; i < 6; ++i) {
        CHECK(abd[i*6+i] > 0.0, "ABD diagonal positive [" + std::to_string(i) + "]");
    }
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 7: Composite Layup Test\n";
    std::cout << "========================================\n";

    test_beam_rectangular();
    test_beam_circular();
    test_beam_hollow();
    test_beam_ibeam();
    test_beam_box();
    test_shell_integration();
    test_variable_thickness();
    test_unidirectional_abd();
    test_cross_ply();
    test_quasi_isotropic();
    test_angle_ply();
    test_ply_stress_recovery();
    test_resultants();
    test_unsymmetric();
    test_composite_section();
    test_section_manager();
    test_abd_matrix_structure();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
