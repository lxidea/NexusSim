/**
 * @file xfem_wave36_test.cpp
 * @brief Wave 36: XFEM Completion Test Suite (6 components, 60 tests)
 *
 * Tests:
 *  7.  XFEMCrackPropagation3D  - 3D crack front advancement
 *  8.  XFEMLayerAdvection      - Shell layer crack tracking
 *  9.  XFEMEnrichment36        - Heaviside + tip enrichment
 *  10. XFEMForceIntegration    - Sub-element integration
 *  11. XFEMCrackDirection      - Crack direction criteria
 *  12. XFEMVelocityUpdate      - Enriched DOF velocity
 */

#include <nexussim/fem/xfem_wave36.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

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
// 7. XFEMCrackPropagation3D Tests
// ============================================================================
void test_crack_propagation_3d() {
    std::cout << "\n--- XFEMCrackPropagation3D Tests ---\n";

    XFEMCrackPropagation3D prop;

    // Test 1: Element cut detection with sign change
    {
        Real phi[] = {1.0, -1.0, 0.5};
        Real psi[] = {-1.0, -0.5, -0.2};
        CHECK(XFEMCrackPropagation3D::is_element_cut(phi, psi, 3),
              "CrackProp3D: element with phi sign change is cut");
    }

    // Test 2: Element not cut (all same sign)
    {
        Real phi[] = {1.0, 0.5, 2.0};
        Real psi[] = {-1.0, -0.5, -0.2};
        CHECK(!XFEMCrackPropagation3D::is_element_cut(phi, psi, 3),
              "CrackProp3D: element without phi sign change is not cut");
    }

    // Test 3: Tip element detection
    {
        Real phi[] = {1.0, -1.0, 0.5, -0.3};
        Real psi[] = {1.0, -0.5, 0.2, -0.1};
        CHECK(XFEMCrackPropagation3D::is_tip_element(phi, psi, 4),
              "CrackProp3D: tip element has both phi and psi sign changes");
    }

    // Test 4: Not a tip element (psi all negative = fully cracked)
    {
        Real phi[] = {1.0, -1.0, 0.5};
        Real psi[] = {-1.0, -0.5, -0.2};
        CHECK(!XFEMCrackPropagation3D::is_tip_element(phi, psi, 3),
              "CrackProp3D: fully cracked element is not tip");
    }

    // Test 5: Advance crack updates psi
    {
        std::vector<CrackNode3D> nodes(5);
        // Line of nodes along x-axis
        for (int i = 0; i < 5; ++i) {
            nodes[i].x = i * 1.0;
            nodes[i].y = 0.0;
            nodes[i].z = 0.0;
            nodes[i].phi = (i < 3) ? -0.1 : 0.1; // crack surface at x~2.5
            nodes[i].psi = (i - 2.0);  // front at x=2
        }

        Real direction[] = {1.0, 0.0, 0.0};
        Real front_pt[] = {2.0, 0.0, 0.0};
        Real tangent[] = {0.0, 1.0, 0.0};

        Real psi_before_3 = nodes[3].psi;
        prop.advance_crack(nodes.data(), 5, direction, 1.5, front_pt, tangent);

        // Node at x=3 had psi=1.0, new front at x=3.5, so new psi for node 3 = 3-3.5 = -0.5
        CHECK(nodes[3].psi < psi_before_3,
              "CrackProp3D: psi decreases after crack advancement");
    }

    // Test 6: Find front nodes within bandwidth
    {
        std::vector<CrackNode3D> nodes(10);
        for (int i = 0; i < 10; ++i) {
            nodes[i].x = i * 0.1;
            nodes[i].y = 0;
            nodes[i].z = 0;
            nodes[i].phi = 0.1;
            nodes[i].psi = (i - 5.0) * 0.1; // psi=0 at node 5
        }

        std::vector<int> front(10);
        int nfront = prop.find_front_nodes(nodes.data(), 10, 0.15, front.data());
        // Nodes with |psi| < 0.15: node 4 (psi=-0.1), node 5 (psi=0), node 6 (psi=0.1)
        CHECK(nfront == 3, "CrackProp3D: found 3 front nodes in bandwidth");
    }

    // Test 7: Crack opening displacement
    {
        Real cod = XFEMCrackPropagation3D::compute_cod(0.001, 0.0, 0.0);
        CHECK_NEAR(cod, 0.002, 1e-10, "CrackProp3D: COD = 2 * enriched displacement");
    }

    // Test 8: COD 3D
    {
        Real cod = XFEMCrackPropagation3D::compute_cod(0.003, 0.004, 0.0);
        // |enriched| = sqrt(9+16)*1e-3 = 0.005, COD = 0.01
        CHECK_NEAR(cod, 0.01, 1e-10, "CrackProp3D: COD 3D magnitude");
    }

    // Test 9: Element not cut when psi all positive (crack hasn't reached)
    {
        Real phi[] = {1.0, -1.0, 0.5};
        Real psi[] = {1.0, 0.5, 2.0}; // All ahead of front
        CHECK(!XFEMCrackPropagation3D::is_element_cut(phi, psi, 3),
              "CrackProp3D: element ahead of crack front is not cut");
    }

    // Test 10: Crack advancement preserves existing phi for far nodes
    {
        std::vector<CrackNode3D> nodes(3);
        nodes[0] = {0, 0, 0, 5.0, 10.0};
        nodes[1] = {0.5, 0, 0, 5.0, 9.5};
        nodes[2] = {100, 0, 0, 50.0, 90.0}; // Far away

        Real direction[] = {1.0, 0.0, 0.0};
        Real front_pt[] = {0, 0, 0};
        Real tangent[] = {0, 1, 0};

        Real phi_far_before = nodes[2].phi;
        prop.advance_crack(nodes.data(), 3, direction, 0.1, front_pt, tangent);
        CHECK_NEAR(nodes[2].phi, phi_far_before, 1e-10,
                   "CrackProp3D: far node phi unchanged");
    }
}

// ============================================================================
// 8. XFEMLayerAdvection Tests
// ============================================================================
void test_layer_advection() {
    std::cout << "\n--- XFEMLayerAdvection Tests ---\n";

    XFEMLayerAdvection advect;

    // Test 11: Initialize shell crack state
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        CHECK(state.nlayers == 4, "LayerAdvect: nlayers = 4");
        CHECK_NEAR(state.total_thickness, 2.0, 1e-10, "LayerAdvect: thickness = 2");
        CHECK(state.cracked_layers == 0, "LayerAdvect: initially 0 cracked");
        CHECK_NEAR(state.phi_layer[0], 1.0, 1e-10, "LayerAdvect: layer 0 intact");
    }

    // Test 12: Advect one layer top-down
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        bool cracked = advect.advect_layer(state, +1);
        CHECK(cracked, "LayerAdvect: layer cracked successfully");
        CHECK(state.cracked_layers == 1, "LayerAdvect: 1 cracked layer");
        CHECK(XFEMLayerAdvection::is_layer_cracked(state, 0),
              "LayerAdvect: layer 0 cracked (top-down)");
    }

    // Test 13: Advect multiple layers
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        advect.advect_layer(state, +1); // layer 0
        advect.advect_layer(state, +1); // layer 1
        advect.advect_layer(state, +1); // layer 2
        CHECK(state.cracked_layers == 3, "LayerAdvect: 3 cracked layers");
        CHECK(!XFEMLayerAdvection::is_fully_cracked(state),
              "LayerAdvect: not fully cracked yet (3/4)");
    }

    // Test 14: Full through-thickness crack
    {
        ShellCrackState state;
        advect.initialize(state, 3, 1.5);
        for (int i = 0; i < 3; ++i) advect.advect_layer(state, +1);
        CHECK(XFEMLayerAdvection::is_fully_cracked(state),
              "LayerAdvect: fully cracked after all layers");
    }

    // Test 15: Cannot crack beyond all layers
    {
        ShellCrackState state;
        advect.initialize(state, 2, 1.0);
        advect.advect_layer(state, +1);
        advect.advect_layer(state, +1);
        bool result = advect.advect_layer(state, +1);
        CHECK(!result, "LayerAdvect: cannot crack beyond nlayers");
    }

    // Test 16: Bottom-up cracking
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        advect.advect_layer(state, -1);
        CHECK(XFEMLayerAdvection::is_layer_cracked(state, 3),
              "LayerAdvect: bottom-up cracks layer 3 first");
        CHECK(!XFEMLayerAdvection::is_layer_cracked(state, 0),
              "LayerAdvect: bottom-up leaves layer 0 intact");
    }

    // Test 17: Ligament fraction
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        CHECK_NEAR(XFEMLayerAdvection::ligament_fraction(state), 1.0, 1e-10,
                   "LayerAdvect: ligament = 1.0 initially");
        advect.advect_layer(state, +1);
        CHECK_NEAR(XFEMLayerAdvection::ligament_fraction(state), 0.75, 1e-10,
                   "LayerAdvect: ligament = 0.75 after 1/4 cracked");
    }

    // Test 18: Stiffness reduction
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        advect.advect_layer(state, +1); // 25% cracked
        auto [membrane, bending] = XFEMLayerAdvection::stiffness_reduction(state);
        CHECK_NEAR(membrane, 0.75, 1e-10, "LayerAdvect: membrane factor = 0.75");
        CHECK_NEAR(bending, 0.421875, 1e-6, "LayerAdvect: bending factor = 0.75^3");
    }

    // Test 19: Crack depth tracks correctly
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        advect.advect_layer(state, +1);
        CHECK_NEAR(state.crack_depth, 0.5, 1e-10,
                   "LayerAdvect: crack depth = 1 layer thickness");
        advect.advect_layer(state, +1);
        CHECK_NEAR(state.crack_depth, 1.0, 1e-10,
                   "LayerAdvect: crack depth = 2 layer thicknesses");
    }

    // Test 20: Crack front z-coordinate (top-down)
    {
        ShellCrackState state;
        advect.initialize(state, 4, 2.0);
        Real z0 = XFEMLayerAdvection::crack_front_z(state, +1);
        CHECK_NEAR(z0, 1.0, 1e-10, "LayerAdvect: initial front at z=+t/2");
        advect.advect_layer(state, +1);
        Real z1 = XFEMLayerAdvection::crack_front_z(state, +1);
        CHECK_NEAR(z1, 0.5, 1e-10, "LayerAdvect: front at z=0.5 after 1 layer");
    }
}

// ============================================================================
// 9. XFEMEnrichment36 Tests
// ============================================================================
void test_enrichment() {
    std::cout << "\n--- XFEMEnrichment36 Tests ---\n";

    XFEMEnrichment36 enrich;

    // Test 21: Heaviside function
    {
        CHECK_NEAR(XFEMEnrichment36::heaviside(1.0), 1.0, 1e-15,
                   "Enrichment: H(1) = 1");
        CHECK_NEAR(XFEMEnrichment36::heaviside(-1.0), -1.0, 1e-15,
                   "Enrichment: H(-1) = -1");
        CHECK_NEAR(XFEMEnrichment36::heaviside(0.0), 0.0, 1e-15,
                   "Enrichment: H(0) = 0");
    }

    // Test 22: Shifted Heaviside
    {
        Real h = XFEMEnrichment36::heaviside_shifted(1.0, -1.0);
        // H(1) - H(-1) = 1 - (-1) = 2
        CHECK_NEAR(h, 2.0, 1e-15, "Enrichment: shifted Heaviside = 2");
    }

    // Test 23: Shifted Heaviside same side
    {
        Real h = XFEMEnrichment36::heaviside_shifted(1.0, 2.0);
        // H(1) - H(2) = 1 - 1 = 0
        CHECK_NEAR(h, 0.0, 1e-15, "Enrichment: shifted Heaviside same side = 0");
    }

    // Test 24: Tip functions at known r, theta
    {
        Real r = 1.0;
        Real theta = 0.0;
        auto F = XFEMEnrichment36::tip_functions(r, theta);
        // F1 = sqrt(1)*sin(0) = 0
        // F2 = sqrt(1)*cos(0) = 1
        // F3 = sqrt(1)*sin(0)*sin(0) = 0
        // F4 = sqrt(1)*cos(0)*sin(0) = 0
        CHECK_NEAR(F[0], 0.0, 1e-15, "TipFunc: F1(1,0) = 0");
        CHECK_NEAR(F[1], 1.0, 1e-10, "TipFunc: F2(1,0) = 1");
        CHECK_NEAR(F[2], 0.0, 1e-15, "TipFunc: F3(1,0) = 0");
        CHECK_NEAR(F[3], 0.0, 1e-15, "TipFunc: F4(1,0) = 0");
    }

    // Test 25: Tip functions at theta = pi
    {
        Real r = 4.0;
        Real theta = M_PI;
        auto F = XFEMEnrichment36::tip_functions(r, theta);
        // F1 = 2 * sin(pi/2) = 2
        // F2 = 2 * cos(pi/2) ~ 0
        // F3 = 2 * sin(pi/2) * sin(pi) ~ 0
        // F4 = 2 * cos(pi/2) * sin(pi) ~ 0
        CHECK_NEAR(F[0], 2.0, 1e-10, "TipFunc: F1(4,pi) = 2");
        CHECK_NEAR(F[1], 0.0, 1e-10, "TipFunc: F2(4,pi) ~ 0");
    }

    // Test 26: Tip functions scale as sqrt(r)
    {
        Real theta = M_PI / 3.0;
        auto F1 = XFEMEnrichment36::tip_functions(1.0, theta);
        auto F4 = XFEMEnrichment36::tip_functions(4.0, theta);
        // F should scale as sqrt(r), so F(4)/F(1) = 2
        CHECK_NEAR(F4[0] / F1[0], 2.0, 1e-10,
                   "TipFunc: sqrt(r) scaling verified");
    }

    // Test 27: Enrich element - cut element
    {
        Real phi[] = {1.0, -1.0, 0.5};
        Real psi[] = {-1.0, -1.0, -1.0}; // All behind front
        EnrichedNode nodes[3];
        int extra = enrich.enrich_element(phi, psi, 3, 2, nodes);
        // Cut element: each node gets ndim=2 extra DOFs
        CHECK(extra == 6, "Enrich: cut tri has 3*2=6 extra DOFs");
        CHECK(nodes[0].type == EnrichmentType::Heaviside,
              "Enrich: nodes are Heaviside-enriched");
    }

    // Test 28: Enrich element - tip element
    {
        Real phi[] = {1.0, -1.0, 0.5, -0.3};
        Real psi[] = {1.0, -0.5, 0.2, -0.1};
        EnrichedNode nodes[4];
        int extra = enrich.enrich_element(phi, psi, 4, 2, nodes);
        // Tip element: each node gets 4*ndim=8 extra DOFs
        CHECK(extra == 32, "Enrich: tip quad has 4*8=32 extra DOFs");
        CHECK(nodes[0].type == EnrichmentType::CrackTip,
              "Enrich: nodes are tip-enriched");
    }

    // Test 29: Enrich element - no enrichment for intact element
    {
        Real phi[] = {1.0, 2.0, 0.5};
        Real psi[] = {1.0, 2.0, 3.0};
        EnrichedNode nodes[3];
        int extra = enrich.enrich_element(phi, psi, 3, 2, nodes);
        CHECK(extra == 0, "Enrich: intact element has 0 extra DOFs");
        CHECK(nodes[0].type == EnrichmentType::None,
              "Enrich: intact nodes are None");
    }

    // Test 30: Enriched shape function
    {
        Real val = XFEMEnrichment36::enriched_shape_heaviside(0.5, 1.0, -1.0);
        // 0.5 * (H(1) - H(-1)) = 0.5 * (1 - (-1)) = 1.0
        CHECK_NEAR(val, 1.0, 1e-15, "Enrich: enriched shape = N*(H-H_I)");
    }

    // Test 31: Enriched shape is zero when same side as node
    {
        Real val = XFEMEnrichment36::enriched_shape_heaviside(0.5, 1.0, 2.0);
        CHECK_NEAR(val, 0.0, 1e-15, "Enrich: enriched shape = 0 same side");
    }

    // Test 32: Extra DOFs per node
    {
        CHECK(XFEMEnrichment36::extra_dofs_per_node(EnrichmentType::Heaviside, 3) == 3,
              "Enrich: Heaviside 3D = 3 extra DOFs");
        CHECK(XFEMEnrichment36::extra_dofs_per_node(EnrichmentType::CrackTip, 3) == 12,
              "Enrich: CrackTip 3D = 12 extra DOFs");
        CHECK(XFEMEnrichment36::extra_dofs_per_node(EnrichmentType::None, 3) == 0,
              "Enrich: None = 0 extra DOFs");
    }

    // Test 33: Polar coordinate conversion
    {
        auto [r, theta] = XFEMEnrichment36::to_polar(2.0, 0.0, 1.0, 0.0, 0.0);
        CHECK_NEAR(r, 1.0, 1e-10, "Polar: r = 1 for point at distance 1 along crack");
        CHECK_NEAR(theta, 0.0, 1e-10, "Polar: theta = 0 along crack");
    }

    // Test 34: Polar coordinate perpendicular to crack
    {
        auto [r, theta] = XFEMEnrichment36::to_polar(1.0, 1.0, 1.0, 0.0, 0.0);
        CHECK_NEAR(r, 1.0, 1e-10, "Polar: r = 1 perpendicular");
        CHECK_NEAR(theta, M_PI/2.0, 1e-10, "Polar: theta = pi/2 above crack");
    }
}

// ============================================================================
// 10. XFEMForceIntegration Tests
// ============================================================================
void test_force_integration() {
    std::cout << "\n--- XFEMForceIntegration Tests ---\n";

    XFEMForceIntegration integ;

    // Test 35: Split triangle with clear sign change
    {
        Real x[] = {0, 1, 0};
        Real y[] = {0, 0, 1};
        Real phi[] = {1, -1, 1}; // Node 1 isolated on negative side

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);
        CHECK(nsub == 3, "SplitTri: 3 sub-triangles for one isolated node");
    }

    // Test 36: No split for uniform sign
    {
        Real x[] = {0, 1, 0};
        Real y[] = {0, 0, 1};
        Real phi[] = {1, 2, 3}; // All positive

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);
        CHECK(nsub == 1, "SplitTri: 1 sub-triangle when no cut");
        CHECK(sub_sides[0] == 1, "SplitTri: positive side");
    }

    // Test 37: Sub-triangle areas sum to original area
    {
        Real x[] = {0, 2, 0};
        Real y[] = {0, 0, 2};
        Real phi[] = {1, -1, 0.5};

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);

        Real total_area = 0;
        for (int t = 0; t < nsub; ++t) {
            total_area += xfem_detail::triangle_area_2d(
                sub_x[t*3], sub_y[t*3], sub_x[t*3+1], sub_y[t*3+1],
                sub_x[t*3+2], sub_y[t*3+2]);
        }
        Real orig_area = xfem_detail::triangle_area_2d(x[0],y[0],x[1],y[1],x[2],y[2]);
        CHECK_NEAR(total_area, orig_area, 1e-10,
                   "SplitTri: sub-triangle areas sum to original");
    }

    // Test 38: Gauss point generation
    {
        Real x[] = {0, 1, 0};
        Real y[] = {0, 0, 1};
        Real phi[] = {1, -1, 1};

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);

        SubGaussPoint gps[12];
        int ngp = integ.generate_gauss_points(sub_x, sub_y, sub_sides, nsub, gps);
        CHECK(ngp == nsub * 3, "GaussGen: 3 points per sub-triangle");
    }

    // Test 39: Integration of constant = area
    {
        Real x[] = {0, 2, 0};
        Real y[] = {0, 0, 2};
        Real phi[] = {1, 2, 3}; // No split

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);

        SubGaussPoint gps[12];
        int ngp = integ.generate_gauss_points(sub_x, sub_y, sub_sides, nsub, gps);

        // Integrate f=1 (should give area = 2)
        std::vector<Real> ones(ngp, 1.0);
        Real area = XFEMForceIntegration::integrate(ones.data(), gps, ngp);
        CHECK_NEAR(area, 2.0, 1e-10, "Integrate: constant 1 gives area");
    }

    // Test 40: Subdomain area computation
    {
        Real x[] = {0, 2, 0};
        Real y[] = {0, 0, 2};
        Real phi[] = {1, -1, 1}; // Split

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);

        SubGaussPoint gps[12];
        int ngp = integ.generate_gauss_points(sub_x, sub_y, sub_sides, nsub, gps);

        Real area_pos = XFEMForceIntegration::subdomain_area(gps, ngp, 1);
        Real area_neg = XFEMForceIntegration::subdomain_area(gps, ngp, -1);

        CHECK(area_pos > 0.0, "SubArea: positive subdomain has area");
        CHECK(area_neg > 0.0, "SubArea: negative subdomain has area");
        Real orig = xfem_detail::triangle_area_2d(x[0],y[0],x[1],y[1],x[2],y[2]);
        CHECK_NEAR(area_pos + area_neg, orig, 1e-8,
                   "SubArea: pos + neg = total area");
    }

    // Test 41: Split respects phi=0 location
    {
        // phi = {1, -1, 0} => zero crossing on edges 0-1 at t=0.5 and 1-2 at t=0.5
        Real x[] = {0, 2, 1};
        Real y[] = {0, 0, 2};
        Real phi[] = {1, -1, 0};

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);
        CHECK(nsub >= 1, "SplitPhi0: split produces sub-triangles");
    }

    // Test 42: Integration accuracy for linear function
    {
        Real x[] = {0, 1, 0};
        Real y[] = {0, 0, 1};
        Real phi[] = {1, 2, 3}; // No split

        Real sub_x[12], sub_y[12];
        int sub_sides[4];
        int nsub = integ.split_triangle(x, y, phi, sub_x, sub_y, sub_sides);

        SubGaussPoint gps[12];
        int ngp = integ.generate_gauss_points(sub_x, sub_y, sub_sides, nsub, gps);

        // Integrate f(x,y) = x + y over triangle [0,1,0]x[0,0,1]
        // Exact: integral of (x+y) over triangle = 1/3
        std::vector<Real> vals(ngp);
        for (int g = 0; g < ngp; ++g) {
            vals[g] = gps[g].x + gps[g].y;
        }
        Real result = XFEMForceIntegration::integrate(vals.data(), gps, ngp);
        CHECK_NEAR(result, 1.0/3.0, 1e-10,
                   "Integrate: linear function exact with 3-pt rule");
    }
}

// ============================================================================
// 11. XFEMCrackDirection Tests
// ============================================================================
void test_crack_direction() {
    std::cout << "\n--- XFEMCrackDirection Tests ---\n";

    XFEMCrackDirection dir;
    dir.set_material(200.0e9, 0.3, 50.0e6, true); // Steel-like

    // Test 43: Pure mode I -> theta = 0
    {
        auto result = dir.compute_direction(50.0e6, 0.0, 0.0);
        CHECK_NEAR(result.theta_c, 0.0, 1e-10,
                   "CrackDir: pure mode I -> theta = 0");
    }

    // Test 44: Pure mode I at fracture toughness -> should grow
    {
        auto result = dir.compute_direction(50.0e6, 0.0, 0.0);
        CHECK(result.should_grow, "CrackDir: K_I = K_Ic -> should grow");
    }

    // Test 45: Below fracture toughness -> should not grow
    {
        auto result = dir.compute_direction(10.0e6, 0.0, 0.0);
        CHECK(!result.should_grow, "CrackDir: K_I < K_Ic -> should not grow");
    }

    // Test 46: Pure mode II angle ~ -70.5 degrees
    {
        auto result = dir.compute_direction(0.0, 50.0e6, 0.0);
        Real expected_deg = -70.5;
        Real expected_rad = expected_deg * M_PI / 180.0;
        CHECK_NEAR(result.theta_c, expected_rad, 0.02,
                   "CrackDir: pure mode II angle ~ -70.5 deg");
    }

    // Test 47: Energy release rate
    {
        Real K_I = 30.0e6;
        auto result = dir.compute_direction(K_I, 0.0, 0.0);
        Real E_prime = dir.E_prime();
        Real G_expected = K_I * K_I / E_prime;
        CHECK_NEAR(result.G, G_expected, G_expected * 1e-10,
                   "CrackDir: G = K_I^2 / E'");
    }

    // Test 48: Mixed mode G
    {
        Real K_I = 30.0e6, K_II = 20.0e6;
        auto result = dir.compute_direction(K_I, K_II, 0.0);
        Real E_prime = dir.E_prime();
        Real G_expected = (K_I*K_I + K_II*K_II) / E_prime;
        CHECK_NEAR(result.G, G_expected, G_expected * 1e-10,
                   "CrackDir: G = (K_I^2 + K_II^2) / E'");
    }

    // Test 49: K from J
    {
        Real J = 1000.0; // J/m^2
        Real K = dir.K_from_J(J);
        Real E_prime = dir.E_prime();
        CHECK_NEAR(K * K / E_prime, J, J * 1e-10,
                   "CrackDir: K_from_J roundtrip");
    }

    // Test 50: Hoop stress at theta=0 for mode I
    {
        Real sigma = XFEMCrackDirection::hoop_stress(1.0, 0.0, 0.0);
        // cos(0) * (1 * cos^2(0) - 0) = 1
        CHECK_NEAR(sigma, 1.0, 1e-10, "CrackDir: hoop stress mode I theta=0");
    }

    // Test 51: Mode mixity
    {
        Real mix = XFEMCrackDirection::mode_mixity(10.0, 5.0);
        CHECK_NEAR(mix, 0.5, 1e-10, "CrackDir: mode mixity = K_II/K_I");
    }

    // Test 52: Global angle conversion
    {
        Real global = XFEMCrackDirection::to_global_angle(0.1, 0.5);
        CHECK_NEAR(global, 0.6, 1e-10, "CrackDir: global = crack + local");
    }
}

// ============================================================================
// 12. XFEMVelocityUpdate Tests
// ============================================================================
void test_velocity_update() {
    std::cout << "\n--- XFEMVelocityUpdate Tests ---\n";

    XFEMVelocityUpdate vel;

    // Test 53: Velocity update with force
    {
        Real v[] = {0.0, 0.0};
        Real f[] = {10.0, 20.0};
        Real m[] = {2.0, 4.0};
        vel.update_velocity(v, f, m, 2, 0.001);
        CHECK_NEAR(v[0], 0.005, 1e-10, "VelUpdate: v += dt*f/m [0]");
        CHECK_NEAR(v[1], 0.005, 1e-10, "VelUpdate: v += dt*f/m [1]");
    }

    // Test 54: Split velocity
    {
        Real std_v[] = {1.0, 2.0};
        Real enr_v[] = {0.1, 0.2};
        Real vp[2], vm[2];
        XFEMVelocityUpdate::split_velocity(std_v, enr_v, 1.0, 2, vp, vm);
        CHECK_NEAR(vp[0], 1.1, 1e-10, "SplitVel: v_plus = std + enr");
        CHECK_NEAR(vm[0], 0.9, 1e-10, "SplitVel: v_minus = std - enr");
        CHECK_NEAR(vp[1], 2.2, 1e-10, "SplitVel: v_plus[1]");
        CHECK_NEAR(vm[1], 1.8, 1e-10, "SplitVel: v_minus[1]");
    }

    // Test 55: Initialize enriched to zero
    {
        Real v[3] = {1, 2, 3};
        XFEMVelocityUpdate::initialize_enriched(v, 3);
        CHECK_NEAR(v[0], 0.0, 1e-15, "InitEnrich: v[0] = 0");
        CHECK_NEAR(v[1], 0.0, 1e-15, "InitEnrich: v[1] = 0");
        CHECK_NEAR(v[2], 0.0, 1e-15, "InitEnrich: v[2] = 0");
    }

    // Test 56: Kinetic energy
    {
        Real v[] = {3.0, 4.0};
        Real m[] = {2.0, 2.0};
        Real ke = XFEMVelocityUpdate::enriched_kinetic_energy(v, m, 2);
        // 0.5 * (2*9 + 2*16) = 0.5 * 50 = 25
        CHECK_NEAR(ke, 25.0, 1e-10, "KE: enriched kinetic energy");
    }

    // Test 57: Position update
    {
        Real x[] = {1.0, 2.0};
        Real v[] = {10.0, -5.0};
        XFEMVelocityUpdate::update_position(x, v, 2, 0.1);
        CHECK_NEAR(x[0], 2.0, 1e-10, "PosUpdate: x += dt*v [0]");
        CHECK_NEAR(x[1], 1.5, 1e-10, "PosUpdate: x += dt*v [1]");
    }

    // Test 58: Crack opening velocity
    {
        Real v[] = {0.5, 0.0};
        Real cov = XFEMVelocityUpdate::crack_opening_velocity(v, 2);
        // COV = 2 * |v| = 2 * 0.5 = 1.0
        CHECK_NEAR(cov, 1.0, 1e-10, "COV: crack opening velocity = 2*|v_enr|");
    }

    // Test 59: COV in 3D
    {
        Real v[] = {0.3, 0.4, 0.0};
        Real cov = XFEMVelocityUpdate::crack_opening_velocity(v, 3);
        CHECK_NEAR(cov, 1.0, 1e-10, "COV 3D: 2*sqrt(0.09+0.16) = 1.0");
    }

    // Test 60: Zero mass doesn't cause division by zero
    {
        Real v[] = {0.0};
        Real f[] = {100.0};
        Real m[] = {0.0}; // Zero mass
        vel.update_velocity(v, f, m, 1, 0.01);
        CHECK_NEAR(v[0], 0.0, 1e-15, "VelUpdate: zero mass -> no update");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 36: XFEM Completion Test Suite ===\n";

    test_crack_propagation_3d();
    test_layer_advection();
    test_enrichment();
    test_force_integration();
    test_crack_direction();
    test_velocity_update();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
