/**
 * @file contact_wave35_test.cpp
 * @brief Wave 35: Advanced Contact Algorithms (int22/int24/int25) Test Suite
 *
 * Tests 10 sub-modules (~10 tests each, ~100 total):
 *  35a: FVM Immersed Boundary (int22)
 *    1. ImmersedBoundaryContact  — Cut cell detection, AABB, volume fraction
 *    2. CutCellGeometry          — Sutherland-Hodgman clipping, sub-volumes
 *    3. ImmersedForce            — Pressure-to-force, balance, distribution
 *  35b: Nitsche Contact (int24)
 *    4. NitscheContact           — Weak-form penalty, gap, force, stability
 *    5. NitscheShellSolid        — Shell-solid coupling, thickness ratio
 *    6. NitschePXFEM             — Enrichment, level-set, interface detection
 *  35c: Full Mortar Contact (int25)
 *    7. MortarContactFull        — D/M matrices, diagonal, gap
 *    8. MortarEdgeToSurface      — Beam-shell projection, penalty force
 *    9. MortarAssembly           — Condensation, positivity, condition
 *   10. MortarThermal            — Heat flux, conservation, dissipation
 */

#include <nexussim/fem/contact_wave35.hpp>
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
// Helper: create a unit cube Euler cell at origin
// ============================================================================
static EulerHexCell make_unit_cube(int id = 0, Real pressure = 100.0) {
    EulerHexCell cell;
    cell.id = id;
    cell.pressure = pressure;
    // Corners of [0,1]^3
    Real coords[8][3] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    for (int i = 0; i < 8; ++i)
        for (int d = 0; d < 3; ++d)
            cell.corners[i][d] = coords[i][d];
    return cell;
}

// ============================================================================
// 1. ImmersedBoundaryContact Tests
// ============================================================================
void test_immersed_boundary_contact() {
    std::cout << "\n=== 35a-1: ImmersedBoundaryContact ===\n";

    ImmersedBoundaryContact ibc(1.0e-12);

    // Test 1: AABB of unit cube
    {
        EulerHexCell cell = make_unit_cube();
        Real bbox_min[3], bbox_max[3];
        ibc.cell_aabb(cell, bbox_min, bbox_max);
        CHECK_NEAR(bbox_min[0], 0.0, 1e-12, "IBC: AABB min x = 0");
        CHECK_NEAR(bbox_max[0], 1.0, 1e-12, "IBC: AABB max x = 1");
        CHECK_NEAR(bbox_max[2], 1.0, 1e-12, "IBC: AABB max z = 1");
    }

    // Test 2: Cell volume of unit cube
    {
        EulerHexCell cell = make_unit_cube();
        Real vol = ibc.cell_volume(cell);
        CHECK_NEAR(vol, 1.0, 1e-12, "IBC: unit cube volume = 1.0");
    }

    // Test 3: Segment-AABB intersection (segment through cell)
    {
        Real p0[3] = {0.5, -0.5, 0.5};
        Real p1[3] = {0.5,  1.5, 0.5};
        Real bbox_min[3] = {0, 0, 0};
        Real bbox_max[3] = {1, 1, 1};
        bool hit = ibc.segment_aabb_intersect(p0, p1, bbox_min, bbox_max);
        CHECK(hit, "IBC: segment through cell intersects AABB");
    }

    // Test 4: Segment-AABB miss
    {
        Real p0[3] = {2.0, 2.0, 0.0};
        Real p1[3] = {3.0, 3.0, 0.0};
        Real bbox_min[3] = {0, 0, 0};
        Real bbox_max[3] = {1, 1, 1};
        bool hit = ibc.segment_aabb_intersect(p0, p1, bbox_min, bbox_max);
        CHECK(!hit, "IBC: distant segment misses AABB");
    }

    // Test 5: Corner classification — plane through center at y=0.5
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, 0.5, 0.5};
        int signs[8];
        int n_above = ibc.classify_corners(cell, normal, point, signs);
        CHECK(n_above == 4, "IBC: plane at y=0.5 splits unit cube 4/4");
    }

    // Test 6: Cut cell detection with a surface cutting through the cell
    {
        EulerHexCell cell = make_unit_cube(0, 100.0);
        // Surface: horizontal line at y=0.5 from x=-0.5 to x=1.5
        Real nodes[4 * 3] = {
            -0.5, 0.5, 0.5,   1.5, 0.5, 0.5,
             1.5, 0.5, 0.5,  -0.5, 0.5, 0.5  // dummy extra nodes
        };
        int segs[2] = {0, 1};  // one segment
        ImmersedSurface surface;
        surface.nodes = nodes;
        surface.segments = segs;
        surface.n_segs = 1;
        surface.n_nodes = 2;

        CutCellInfo info;
        int n_cut = ibc.detect_cut_cells(surface, &cell, 1, &info);
        CHECK(n_cut == 1, "IBC: surface through cell detects 1 cut cell");
        CHECK(info.is_cut, "IBC: cut cell info reports is_cut = true");
    }

    // Test 7: Volume fraction for symmetric cut
    {
        EulerHexCell cell = make_unit_cube(0, 100.0);
        Real nodes[2 * 3] = { -0.5, 0.5, 0.5,  1.5, 0.5, 0.5 };
        int segs[2] = {0, 1};
        ImmersedSurface surface;
        surface.nodes = nodes;
        surface.segments = segs;
        surface.n_segs = 1;
        surface.n_nodes = 2;

        CutCellInfo info;
        ibc.detect_cut_cells(surface, &cell, 1, &info);
        // Symmetric cut: volume_fraction should be close to 0.5
        CHECK_NEAR(info.volume_fraction, 0.5, 0.1, "IBC: symmetric cut gives ~50% volume fraction");
    }

    // Test 8: No cut when surface is outside cell
    {
        EulerHexCell cell = make_unit_cube(0, 100.0);
        Real nodes[2 * 3] = { 5.0, 5.0, 5.0,  6.0, 5.0, 5.0 };
        int segs[2] = {0, 1};
        ImmersedSurface surface;
        surface.nodes = nodes;
        surface.segments = segs;
        surface.n_segs = 1;
        surface.n_nodes = 2;

        CutCellInfo info;
        int n_cut = ibc.detect_cut_cells(surface, &cell, 1, &info);
        CHECK(n_cut == 0, "IBC: surface outside cell gives 0 cut cells");
    }

    // Test 9: Wetted area is positive for cut cells
    {
        EulerHexCell cell = make_unit_cube(0, 100.0);
        Real nodes[2 * 3] = { -0.5, 0.5, 0.5,  1.5, 0.5, 0.5 };
        int segs[2] = {0, 1};
        ImmersedSurface surface;
        surface.nodes = nodes;
        surface.segments = segs;
        surface.n_segs = 1;
        surface.n_nodes = 2;

        CutCellInfo info;
        ibc.detect_cut_cells(surface, &cell, 1, &info);
        CHECK(info.wetted_area > 0.0, "IBC: wetted area is positive for cut cell");
    }

    // Test 10: Cell volume scaled correctly
    {
        EulerHexCell cell;
        cell.id = 0; cell.pressure = 0.0;
        for (int i = 0; i < 8; ++i)
            for (int d = 0; d < 3; ++d)
                cell.corners[i][d] = make_unit_cube().corners[i][d] * 2.0;
        Real vol = ibc.cell_volume(cell);
        CHECK_NEAR(vol, 8.0, 1e-12, "IBC: doubled cube volume = 8.0");
    }
}

// ============================================================================
// 2. CutCellGeometry Tests
// ============================================================================
void test_cut_cell_geometry() {
    std::cout << "\n=== 35a-2: CutCellGeometry ===\n";

    CutCellGeometry ccg;

    // Test 1: Clip cell — no cut (all corners above plane)
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, -0.5, 0.5};  // Plane below cell
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK_NEAR(vl, 1.0, 1e-10, "CCG: plane below cell gives full left volume");
        CHECK_NEAR(vr, 0.0, 1e-10, "CCG: plane below cell gives zero right volume");
        CHECK_NEAR(wa, 0.0, 1e-10, "CCG: no wetted area when no cut");
    }

    // Test 2: Clip cell — symmetric cut at y=0.5
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, 0.5, 0.5};
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK_NEAR(vl + vr, 1.0, 1e-10, "CCG: volume conservation (left + right = total)");
    }

    // Test 3: Sub-volumes approximately equal for centered plane
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, 0.5, 0.5};
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK_NEAR(vl, vr, 0.15, "CCG: symmetric plane gives approx equal sub-volumes");
    }

    // Test 4: Wetted area is positive for cut
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, 0.5, 0.5};
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK(wa > 0.0, "CCG: positive wetted area for cut cell");
    }

    // Test 5: Polygon area of a unit square
    {
        Real verts[4][3] = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
        Real area = ccg.polygon_area(verts, 4);
        CHECK_NEAR(area, 1.0, 1e-10, "CCG: unit square polygon area = 1.0");
    }

    // Test 6: Polygon area of a triangle
    {
        Real verts[3][3] = {{0,0,0}, {2,0,0}, {0,2,0}};
        Real area = ccg.polygon_area(verts, 3);
        CHECK_NEAR(area, 2.0, 1e-10, "CCG: right triangle area = 2.0");
    }

    // Test 7: Volume conservation for off-center cut
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.5, 0.25, 0.5};
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK_NEAR(vl + vr, 1.0, 1e-10, "CCG: volume conservation for off-center cut");
    }

    // Test 8: Diagonal plane cut conserves volume
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};
        Real point[3] = {0.5, 0.5, 0.5};
        Real vl, vr, wa;
        ccg.clip_cell(cell.corners, normal, point, vl, vr, wa);
        CHECK_NEAR(vl + vr, 1.0, 1e-10, "CCG: diagonal cut conserves total volume");
    }

    // Test 9: Clip polygon keeps all vertices when fully inside
    {
        Real in_verts[4][3] = {{0.2, 0.2, 0.0}, {0.8, 0.2, 0.0},
                                {0.8, 0.8, 0.0}, {0.2, 0.8, 0.0}};
        Real out_verts[CutCellGeometry::MAX_POLY_VERTS][3];
        Real normal[3] = {0.0, 1.0, 0.0};
        Real point[3] = {0.0, 0.0, 0.0};  // Plane at y=0, all verts above
        int n_out = ccg.clip_polygon(in_verts, 4, normal, point, out_verts);
        CHECK(n_out == 4, "CCG: fully inside polygon retains all 4 vertices");
    }

    // Test 10: Wetted area scales with cut fraction
    {
        EulerHexCell cell = make_unit_cube();
        Real normal[3] = {0.0, 1.0, 0.0};

        Real point1[3] = {0.5, 0.5, 0.5};  // Center cut
        Real point2[3] = {0.5, 0.25, 0.5}; // Off-center cut
        Real vl1, vr1, wa1, vl2, vr2, wa2;
        ccg.clip_cell(cell.corners, normal, point1, vl1, vr1, wa1);
        ccg.clip_cell(cell.corners, normal, point2, vl2, vr2, wa2);
        // Center cut should have largest wetted area
        CHECK(wa1 >= wa2, "CCG: centered cut has largest wetted area");
    }
}

// ============================================================================
// 3. ImmersedForce Tests
// ============================================================================
void test_immersed_force() {
    std::cout << "\n=== 35a-3: ImmersedForce ===\n";

    ImmersedForce imf;

    // Test 1: Segment normal direction (horizontal segment -> vertical normal)
    {
        Real p0[3] = {0.0, 0.0, 0.0};
        Real p1[3] = {1.0, 0.0, 0.0};
        Real normal[3];
        imf.segment_normal(p0, p1, normal);
        CHECK_NEAR(std::abs(normal[1]), 1.0, 1e-10, "ImmF: horizontal segment has y-normal");
        CHECK_NEAR(normal[0], 0.0, 1e-10, "ImmF: horizontal segment normal x=0");
    }

    // Test 2: Segment length
    {
        Real p0[3] = {0.0, 0.0, 0.0};
        Real p1[3] = {3.0, 4.0, 0.0};
        Real len = imf.segment_length(p0, p1);
        CHECK_NEAR(len, 5.0, 1e-10, "ImmF: segment length = 5 for 3-4-5 triangle");
    }

    // Test 3: Force with single cut cell — nonzero force
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = true;
        info.volume_fraction = 0.5;
        info.wetted_area = 1.0;

        Real pressure[1] = {1000.0};
        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};
        Real forces[2 * 3];

        imf.compute_immersed_force(&info, 1, pressure, nodes, segs, 1, 2, forces);

        Real total[3];
        imf.total_surface_force(forces, 2, total);
        Real mag = std::sqrt(total[0]*total[0] + total[1]*total[1] + total[2]*total[2]);
        CHECK(mag > 0.0, "ImmF: nonzero force from pressure on cut cell");
    }

    // Test 4: Zero pressure -> zero force
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = true;
        info.volume_fraction = 0.5;
        info.wetted_area = 1.0;

        Real pressure[1] = {0.0};
        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};
        Real forces[2 * 3];

        imf.compute_immersed_force(&info, 1, pressure, nodes, segs, 1, 2, forces);

        Real total[3];
        imf.total_surface_force(forces, 2, total);
        Real mag = std::sqrt(total[0]*total[0] + total[1]*total[1] + total[2]*total[2]);
        CHECK_NEAR(mag, 0.0, 1e-15, "ImmF: zero pressure gives zero force");
    }

    // Test 5: Force direction follows segment normal
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = true;
        info.volume_fraction = 0.5;
        info.wetted_area = 1.0;

        Real pressure[1] = {1000.0};
        // Horizontal segment along x -> normal in y direction
        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};
        Real forces[2 * 3];

        imf.compute_immersed_force(&info, 1, pressure, nodes, segs, 1, 2, forces);

        Real total[3];
        imf.total_surface_force(forces, 2, total);
        // Force should be primarily in y direction
        CHECK(std::abs(total[1]) > std::abs(total[0]),
              "ImmF: force direction follows segment normal");
    }

    // Test 6: Force splits equally between segment endpoints
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = true;
        info.volume_fraction = 0.5;
        info.wetted_area = 1.0;

        Real pressure[1] = {1000.0};
        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};
        Real forces[2 * 3];

        imf.compute_immersed_force(&info, 1, pressure, nodes, segs, 1, 2, forces);
        // Node 0 force should equal node 1 force
        for (int d = 0; d < 3; ++d) {
            CHECK_NEAR(forces[0*3+d], forces[1*3+d], 1e-10,
                      "ImmF: force equally distributed to endpoints");
        }
    }

    // Test 7: Non-cut cell contributes no force
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = false;
        info.volume_fraction = 1.0;
        info.wetted_area = 0.0;

        Real pressure[1] = {1000.0};
        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};
        Real forces[2 * 3];

        imf.compute_immersed_force(&info, 1, pressure, nodes, segs, 1, 2, forces);
        Real total[3];
        imf.total_surface_force(forces, 2, total);
        Real mag = std::sqrt(total[0]*total[0] + total[1]*total[1] + total[2]*total[2]);
        CHECK_NEAR(mag, 0.0, 1e-15, "ImmF: non-cut cell gives zero force");
    }

    // Test 8: Force scales with pressure
    {
        CutCellInfo info;
        info.cell_id = 0;
        info.is_cut = true;
        info.volume_fraction = 0.5;
        info.wetted_area = 1.0;

        Real nodes[2 * 3] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        int segs[2] = {0, 1};

        Real p1[1] = {500.0};
        Real p2[1] = {1000.0};
        Real f1[2*3], f2[2*3];

        imf.compute_immersed_force(&info, 1, p1, nodes, segs, 1, 2, f1);
        imf.compute_immersed_force(&info, 1, p2, nodes, segs, 1, 2, f2);

        Real t1[3], t2[3];
        imf.total_surface_force(f1, 2, t1);
        imf.total_surface_force(f2, 2, t2);

        Real mag1 = std::sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
        Real mag2 = std::sqrt(t2[0]*t2[0] + t2[1]*t2[1] + t2[2]*t2[2]);
        CHECK_NEAR(mag2 / mag1, 2.0, 1e-10, "ImmF: force scales linearly with pressure");
    }

    // Test 9: Total surface force sums correctly
    {
        Real forces[3 * 3] = {1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0};
        Real total[3];
        imf.total_surface_force(forces, 3, total);
        CHECK_NEAR(total[0], 12.0, 1e-12, "ImmF: total force x = 1+4+7 = 12");
        CHECK_NEAR(total[1], 15.0, 1e-12, "ImmF: total force y = 2+5+8 = 15");
        CHECK_NEAR(total[2], 18.0, 1e-12, "ImmF: total force z = 3+6+9 = 18");
    }
}

// ============================================================================
// 4. NitscheContact Tests
// ============================================================================
void test_nitsche_contact() {
    std::cout << "\n=== 35b-1: NitscheContact ===\n";

    NitscheContact nc;

    // Two planar quad elements facing each other
    NitscheElement master, slave;
    master.E = 2.0e11; master.nu = 0.3; master.id = 0;
    slave.E = 2.0e11; slave.nu = 0.3; slave.id = 1;
    // Master at z=0 plane
    master.nodes[0][0] = 0; master.nodes[0][1] = 0; master.nodes[0][2] = 0;
    master.nodes[1][0] = 1; master.nodes[1][1] = 0; master.nodes[1][2] = 0;
    master.nodes[2][0] = 1; master.nodes[2][1] = 1; master.nodes[2][2] = 0;
    master.nodes[3][0] = 0; master.nodes[3][1] = 1; master.nodes[3][2] = 0;
    // Slave at z=0.01 (small gap)
    for (int i = 0; i < 4; ++i) {
        slave.nodes[i][0] = master.nodes[i][0];
        slave.nodes[i][1] = master.nodes[i][1];
        slave.nodes[i][2] = 0.01;
    }

    // Test 1: Gap computation
    {
        Real normal[3] = {0, 0, 1};
        Real gap = nc.compute_gap(master, slave, normal);
        CHECK_NEAR(gap, 0.01, 1e-10, "Nitsche: gap = 0.01 for separated elements");
    }

    // Test 2: Average normal stress
    {
        Real sigma = nc.average_normal_stress(master, slave, -0.01, 0.1);
        CHECK(sigma < 0.0, "Nitsche: compressive stress for negative gap");
    }

    // Test 3: Stability check for symmetric Nitsche (theta = -1)
    {
        NitscheConfig config;
        config.gamma_N = 1.0e15;  // Very large
        config.theta = -1;
        config.h_element = 0.1;
        bool stable = nc.is_stable(master, slave, config);
        CHECK(stable, "Nitsche: large gamma_N is stable for symmetric");
    }

    // Test 4: Unsymmetric is always stable
    {
        NitscheConfig config;
        config.gamma_N = 1.0;  // Tiny
        config.theta = 0;
        config.h_element = 0.1;
        bool stable = nc.is_stable(master, slave, config);
        CHECK(stable, "Nitsche: unsymmetric (theta=0) always stable");
    }

    // Test 5: Skew-symmetric is always stable
    {
        NitscheConfig config;
        config.gamma_N = 1.0;
        config.theta = 1;
        config.h_element = 0.1;
        bool stable = nc.is_stable(master, slave, config);
        CHECK(stable, "Nitsche: skew-symmetric (theta=1) always stable");
    }

    // Test 6: Force magnitude is nonzero for contacting elements
    {
        // Make elements overlap
        NitscheElement slave_pen = slave;
        for (int i = 0; i < 4; ++i) slave_pen.nodes[i][2] = -0.005; // penetrating

        NitscheConfig config;
        config.gamma_N = 1.0e12;
        config.theta = -1;
        config.h_element = 0.1;

        Real forces[8][3];
        Real mag = nc.compute_nitsche_forces(master, slave_pen, config, forces);
        CHECK(mag > 0.0, "Nitsche: nonzero force for penetrating elements");
    }

    // Test 7: Action-reaction (sum of all forces ~ 0)
    {
        NitscheElement slave_pen = slave;
        for (int i = 0; i < 4; ++i) slave_pen.nodes[i][2] = -0.005;

        NitscheConfig config;
        config.gamma_N = 1.0e12;
        config.theta = -1;
        config.h_element = 0.1;

        Real forces[8][3];
        nc.compute_nitsche_forces(master, slave_pen, config, forces);

        Real total[3] = {0, 0, 0};
        for (int i = 0; i < 8; ++i)
            for (int d = 0; d < 3; ++d)
                total[d] += forces[i][d];

        Real balance = std::sqrt(total[0]*total[0] + total[1]*total[1] + total[2]*total[2]);
        CHECK_NEAR(balance, 0.0, 1e-6, "Nitsche: force balance (action = reaction)");
    }

    // Test 8: gamma_N scaling — larger gamma gives larger force
    {
        NitscheElement slave_pen = slave;
        for (int i = 0; i < 4; ++i) slave_pen.nodes[i][2] = -0.005;

        NitscheConfig config1, config2;
        config1.gamma_N = 1.0e10; config1.theta = 0; config1.h_element = 0.1;
        config2.gamma_N = 2.0e10; config2.theta = 0; config2.h_element = 0.1;

        Real f1[8][3], f2[8][3];
        Real mag1 = nc.compute_nitsche_forces(master, slave_pen, config1, f1);
        Real mag2 = nc.compute_nitsche_forces(master, slave_pen, config2, f2);
        CHECK(mag2 > mag1, "Nitsche: larger gamma_N gives larger force");
    }

    // Test 9: Larger penetration gives larger force than smaller penetration
    {
        NitscheConfig config;
        config.gamma_N = 1.0e12; config.theta = 0; config.h_element = 0.1;

        NitscheElement slave_sep = slave;
        for (int i = 0; i < 4; ++i) slave_sep.nodes[i][2] = -0.005;  // small penetration
        NitscheElement slave_pen = slave;
        for (int i = 0; i < 4; ++i) slave_pen.nodes[i][2] = -0.02;  // large penetration

        Real f_sep[8][3], f_pen[8][3];
        Real mag_sep = nc.compute_nitsche_forces(master, slave_sep, config, f_sep);
        Real mag_pen = nc.compute_nitsche_forces(master, slave_pen, config, f_pen);
        CHECK(mag_pen > mag_sep, "Nitsche: deeper penetration gives larger force");
    }
}

// ============================================================================
// 5. NitscheShellSolid Tests
// ============================================================================
void test_nitsche_shell_solid() {
    std::cout << "\n=== 35b-2: NitscheShellSolid ===\n";

    NitscheShellSolid nss;

    NitscheShellElement shell;
    shell.thickness = 0.01;
    shell.E = 2.0e11; shell.nu = 0.3;
    shell.nodes[0][0] = 0; shell.nodes[0][1] = 0; shell.nodes[0][2] = 0;
    shell.nodes[1][0] = 1; shell.nodes[1][1] = 0; shell.nodes[1][2] = 0;
    shell.nodes[2][0] = 1; shell.nodes[2][1] = 1; shell.nodes[2][2] = 0;
    shell.nodes[3][0] = 0; shell.nodes[3][1] = 1; shell.nodes[3][2] = 0;

    NitscheSolidElement solid;
    solid.E = 2.0e11; solid.nu = 0.3;
    // Bottom face matches shell, top face at z=0.1
    Real base[4][3] = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
    for (int i = 0; i < 4; ++i) {
        for (int d = 0; d < 3; ++d) {
            solid.nodes[i][d] = base[i][d];
            solid.nodes[i+4][d] = base[i][d];
        }
        solid.nodes[i+4][2] = 0.1;
    }

    NitscheConfig config;
    config.gamma_N = 1.0e10;
    config.theta = 0;
    config.h_element = 0.1;

    // Test 1: Coupling stiffness is positive
    {
        Real K;
        nss.couple_shell_solid(shell, solid, config, K);
        CHECK(K > 0.0, "ShellSolid: coupling stiffness is positive");
    }

    // Test 2: Coupling energy is zero when coincident
    {
        Real K;
        Real energy = nss.couple_shell_solid(shell, solid, config, K);
        CHECK_NEAR(energy, 0.0, 1e-6, "ShellSolid: zero energy when coincident");
    }

    // Test 3: Coupling energy increases with gap
    {
        NitscheShellElement shell_off = shell;
        for (int i = 0; i < 4; ++i) shell_off.nodes[i][2] = 0.05;

        Real K1, K2;
        Real e1 = nss.couple_shell_solid(shell, solid, config, K1);
        Real e2 = nss.couple_shell_solid(shell_off, solid, config, K2);
        CHECK(e2 > e1, "ShellSolid: energy increases with gap");
    }

    // Test 4: Thickness ratio
    {
        Real ratio = nss.thickness_ratio(shell, solid);
        // shell.thickness = 0.01, solid thickness ~ 0.1
        CHECK_NEAR(ratio, 0.1, 1e-10, "ShellSolid: thickness ratio = 0.01/0.1 = 0.1");
    }

    // Test 5: Stiffness scales with gamma_N
    {
        NitscheConfig config2 = config;
        config2.gamma_N = 2.0e10;
        Real K1, K2;
        nss.couple_shell_solid(shell, solid, config, K1);
        nss.couple_shell_solid(shell, solid, config2, K2);
        CHECK_NEAR(K2 / K1, 2.0, 1e-10, "ShellSolid: stiffness scales with gamma_N");
    }

    // Test 6: Stiffness scales inversely with h_element
    {
        NitscheConfig config2 = config;
        config2.h_element = 0.2;  // double h
        Real K1, K2;
        nss.couple_shell_solid(shell, solid, config, K1);
        nss.couple_shell_solid(shell, solid, config2, K2);
        CHECK_NEAR(K2 / K1, 0.5, 1e-10, "ShellSolid: stiffness inversely proportional to h");
    }

    // Test 7: Different moduli give different stiffnesses
    {
        NitscheSolidElement solid_soft = solid;
        solid_soft.E = 1.0e11;  // Half stiffness
        Real K1, K2;
        nss.couple_shell_solid(shell, solid, config, K1);
        nss.couple_shell_solid(shell, solid_soft, config, K2);
        // Same K since coupling depends on gamma/h*area, not E directly
        CHECK_NEAR(K1, K2, 1e-3, "ShellSolid: coupling K depends on gamma/h, not E");
    }

    // Test 8: Shell area influences coupling stiffness
    {
        NitscheShellElement shell_big = shell;
        for (int i = 0; i < 4; ++i) {
            shell_big.nodes[i][0] *= 2.0;
            shell_big.nodes[i][1] *= 2.0;
        }
        Real K1, K2;
        nss.couple_shell_solid(shell, solid, config, K1);
        nss.couple_shell_solid(shell_big, solid, config, K2);
        CHECK(K2 > K1, "ShellSolid: larger shell area increases coupling stiffness");
    }
}

// ============================================================================
// 6. NitschePXFEM Tests
// ============================================================================
void test_nitsche_pxfem() {
    std::cout << "\n=== 35b-3: NitschePXFEM ===\n";

    NitschePXFEM pxfem;

    // Test 1: Element cut detection (sign change in phi)
    {
        Real phi_cut[4] = {1.0, -1.0, -1.0, 1.0};
        CHECK(pxfem.is_element_cut(phi_cut), "PXFEM: sign change detects cut element");
    }

    // Test 2: No cut when all same sign
    {
        Real phi_pos[4] = {1.0, 2.0, 0.5, 3.0};
        CHECK(!pxfem.is_element_cut(phi_pos), "PXFEM: all positive => not cut");
    }

    // Test 3: No cut when all negative
    {
        Real phi_neg[4] = {-1.0, -2.0, -0.5, -3.0};
        CHECK(!pxfem.is_element_cut(phi_neg), "PXFEM: all negative => not cut");
    }

    // Test 4: Enrichment function at node = 0
    {
        Real val = pxfem.enrichment_function(1.5, 1.5);
        CHECK_NEAR(val, 0.0, 1e-15, "PXFEM: enrichment = 0 at its own node");
    }

    // Test 5: Enrichment function away from node
    {
        Real val = pxfem.enrichment_function(2.0, 1.0);
        // |2.0| - |1.0| = 1.0
        CHECK_NEAR(val, 1.0, 1e-15, "PXFEM: enrichment = |phi(x)| - |phi(x_j)|");
    }

    // Test 6: Interface point on edge with linear interpolation
    {
        Real x_i[3] = {0.0, 0.0, 0.0};
        Real x_j[3] = {1.0, 0.0, 0.0};
        Real phi_i = 1.0, phi_j = -1.0;
        Real x_int[3];
        pxfem.interface_point_on_edge(x_i, x_j, phi_i, phi_j, x_int);
        CHECK_NEAR(x_int[0], 0.5, 1e-12, "PXFEM: interface at midpoint for symmetric phi");
    }

    // Test 7: Interface point with asymmetric phi
    {
        Real x_i[3] = {0.0, 0.0, 0.0};
        Real x_j[3] = {1.0, 0.0, 0.0};
        Real phi_i = 3.0, phi_j = -1.0;
        Real x_int[3];
        pxfem.interface_point_on_edge(x_i, x_j, phi_i, phi_j, x_int);
        CHECK_NEAR(x_int[0], 0.75, 1e-12, "PXFEM: interface at 3/4 for phi_i=3, phi_j=-1");
    }

    // Test 8: Enrich and couple — returns enriched DOF count
    {
        PXFEMCutElement elem;
        elem.id = 0; elem.is_cut = true;
        elem.nodes[0][0] = 0; elem.nodes[0][1] = 0; elem.nodes[0][2] = 0;
        elem.nodes[1][0] = 1; elem.nodes[1][1] = 0; elem.nodes[1][2] = 0;
        elem.nodes[2][0] = 1; elem.nodes[2][1] = 1; elem.nodes[2][2] = 0;
        elem.nodes[3][0] = 0; elem.nodes[3][1] = 1; elem.nodes[3][2] = 0;
        elem.phi[0] = 1.0; elem.phi[1] = -1.0;
        elem.phi[2] = -1.0; elem.phi[3] = 1.0;
        for (int i = 0; i < 4; ++i) elem.enriched_dofs[i] = 0.0;

        NitscheConfig config;
        config.gamma_N = 1.0e6; config.theta = 0; config.h_element = 1.0;
        Real phi_field[4] = {1.0, -1.0, -1.0, 1.0};
        Real enriched_K[1];

        int n_dofs = pxfem.enrich_and_couple(&elem, 1, phi_field, config, enriched_K);
        CHECK(n_dofs == 4, "PXFEM: 4 enriched DOFs for one cut quad");
    }

    // Test 9: Enriched stiffness is positive for cut element
    {
        PXFEMCutElement elem;
        elem.id = 0; elem.is_cut = true;
        elem.nodes[0][0] = 0; elem.nodes[0][1] = 0; elem.nodes[0][2] = 0;
        elem.nodes[1][0] = 1; elem.nodes[1][1] = 0; elem.nodes[1][2] = 0;
        elem.nodes[2][0] = 1; elem.nodes[2][1] = 1; elem.nodes[2][2] = 0;
        elem.nodes[3][0] = 0; elem.nodes[3][1] = 1; elem.nodes[3][2] = 0;
        elem.phi[0] = 1.0; elem.phi[1] = -1.0;
        elem.phi[2] = -1.0; elem.phi[3] = 1.0;
        for (int i = 0; i < 4; ++i) elem.enriched_dofs[i] = 0.0;

        NitscheConfig config;
        config.gamma_N = 1.0e6; config.theta = 0; config.h_element = 1.0;
        Real phi_field[4] = {1.0, -1.0, -1.0, 1.0};
        Real enriched_K[1];

        pxfem.enrich_and_couple(&elem, 1, phi_field, config, enriched_K);
        CHECK(enriched_K[0] > 0.0, "PXFEM: enriched stiffness is positive");
    }

    // Test 10: Non-cut element gets zero enriched stiffness
    {
        PXFEMCutElement elem;
        elem.id = 0; elem.is_cut = false;
        for (int i = 0; i < 4; ++i) {
            elem.phi[i] = 1.0;
            elem.enriched_dofs[i] = 0.0;
            for (int d = 0; d < 3; ++d) elem.nodes[i][d] = 0.0;
        }

        NitscheConfig config;
        config.gamma_N = 1.0e6; config.theta = 0; config.h_element = 1.0;
        Real phi_field[4] = {1.0, 1.0, 1.0, 1.0};
        Real enriched_K[1];

        int n_dofs = pxfem.enrich_and_couple(&elem, 1, phi_field, config, enriched_K);
        CHECK(n_dofs == 0, "PXFEM: no enriched DOFs for non-cut element");
        CHECK_NEAR(enriched_K[0], 0.0, 1e-15, "PXFEM: zero enriched K for non-cut element");
    }
}

// ============================================================================
// 7. MortarContactFull Tests
// ============================================================================
void test_mortar_contact_full() {
    std::cout << "\n=== 35c-1: MortarContactFull ===\n";

    MortarContactFull mcf;

    MortarSegment master_seg, slave_seg;
    master_seg.id = 0; slave_seg.id = 0;
    // Master at z=0
    master_seg.nodes[0][0] = 0; master_seg.nodes[0][1] = 0; master_seg.nodes[0][2] = 0;
    master_seg.nodes[1][0] = 1; master_seg.nodes[1][1] = 0; master_seg.nodes[1][2] = 0;
    master_seg.nodes[2][0] = 1; master_seg.nodes[2][1] = 1; master_seg.nodes[2][2] = 0;
    master_seg.nodes[3][0] = 0; master_seg.nodes[3][1] = 1; master_seg.nodes[3][2] = 0;
    // Slave at z=0.01
    for (int i = 0; i < 4; ++i) {
        for (int d = 0; d < 3; ++d) slave_seg.nodes[i][d] = master_seg.nodes[i][d];
        slave_seg.nodes[i][2] = 0.01;
    }

    // Test 1: Shape functions sum to 1
    {
        Real N[4];
        MortarContactFull::shape_functions_quad(0.3, -0.2, N);
        Real sum = N[0] + N[1] + N[2] + N[3];
        CHECK_NEAR(sum, 1.0, 1e-12, "Mortar: shape functions sum to 1.0");
    }

    // Test 2: Shape functions at center
    {
        Real N[4];
        MortarContactFull::shape_functions_quad(0.0, 0.0, N);
        CHECK_NEAR(N[0], 0.25, 1e-12, "Mortar: N1(0,0) = 0.25");
        CHECK_NEAR(N[1], 0.25, 1e-12, "Mortar: N2(0,0) = 0.25");
    }

    // Test 3: Segment area of unit square
    {
        Real area = mcf.segment_area(master_seg);
        CHECK_NEAR(area, 1.0, 1e-10, "Mortar: unit square segment area = 1.0");
    }

    // Test 4: D matrix is non-negative (diagonal dual basis)
    {
        Real D[4], M[16];
        mcf.compute_mortar_matrices(&master_seg, &slave_seg, 1, 1, D, M);
        bool all_pos = true;
        for (int i = 0; i < 4; ++i) {
            if (D[i] < -1e-15) all_pos = false;
        }
        CHECK(all_pos, "Mortar: D matrix entries are non-negative");
    }

    // Test 5: D is diagonal (by construction in our storage)
    {
        Real D[4], M[16];
        mcf.compute_mortar_matrices(&master_seg, &slave_seg, 1, 1, D, M);
        CHECK(mcf.is_diagonal(D, 4), "Mortar: D matrix passes diagonal check");
    }

    // Test 6: M matrix has positive entries for overlapping segments
    {
        Real D[4], M[16];
        mcf.compute_mortar_matrices(&master_seg, &slave_seg, 1, 1, D, M);
        Real M_sum = 0.0;
        for (int i = 0; i < 16; ++i) M_sum += M[i];
        CHECK(M_sum > 0.0, "Mortar: M matrix has positive sum for overlapping segments");
    }

    // Test 7: Mortar gap computation
    {
        Real normal[3] = {0, 0, 1};
        Real gap = mcf.mortar_gap(master_seg, slave_seg, normal);
        CHECK_NEAR(gap, 0.01, 1e-10, "Mortar: gap = 0.01 between parallel segments");
    }

    // Test 8: Segment centroid
    {
        Real centroid[3];
        mcf.segment_centroid(master_seg, centroid);
        CHECK_NEAR(centroid[0], 0.5, 1e-12, "Mortar: centroid x = 0.5");
        CHECK_NEAR(centroid[1], 0.5, 1e-12, "Mortar: centroid y = 0.5");
    }

    // Test 9: D entries sum to segment area
    {
        Real D[4], M[16];
        mcf.compute_mortar_matrices(&master_seg, &slave_seg, 1, 1, D, M);
        Real D_sum = D[0] + D[1] + D[2] + D[3];
        Real area = mcf.segment_area(slave_seg);
        CHECK_NEAR(D_sum, area, 0.01, "Mortar: D entries sum to segment area");
    }

    // Test 10: Shape function at corner recovers Kronecker delta
    {
        Real N[4];
        MortarContactFull::shape_functions_quad(-1.0, -1.0, N);
        CHECK_NEAR(N[0], 1.0, 1e-12, "Mortar: N1(-1,-1) = 1 (Kronecker delta)");
        CHECK_NEAR(N[1], 0.0, 1e-12, "Mortar: N2(-1,-1) = 0 (Kronecker delta)");
    }
}

// ============================================================================
// 8. MortarEdgeToSurface Tests
// ============================================================================
void test_mortar_edge_to_surface() {
    std::cout << "\n=== 35c-2: MortarEdgeToSurface ===\n";

    MortarEdgeToSurface mes(1.0e6);

    MortarSegment shell_seg;
    shell_seg.id = 0;
    shell_seg.nodes[0][0] = 0; shell_seg.nodes[0][1] = 0; shell_seg.nodes[0][2] = 0;
    shell_seg.nodes[1][0] = 2; shell_seg.nodes[1][1] = 0; shell_seg.nodes[1][2] = 0;
    shell_seg.nodes[2][0] = 2; shell_seg.nodes[2][1] = 2; shell_seg.nodes[2][2] = 0;
    shell_seg.nodes[3][0] = 0; shell_seg.nodes[3][1] = 2; shell_seg.nodes[3][2] = 0;

    // Test 1: Projection onto surface
    {
        Real point[3] = {1.0, 1.0, 0.5};
        Real proj[3], normal[3];
        Real gap;
        mes.project_to_segment(point, shell_seg, proj, gap, normal);
        CHECK_NEAR(gap, 0.5, 1e-10, "EdgeSurf: gap = 0.5 for point above z=0 plane");
    }

    // Test 2: Normal direction
    {
        Real point[3] = {1.0, 1.0, 0.5};
        Real proj[3], normal[3];
        Real gap;
        mes.project_to_segment(point, shell_seg, proj, gap, normal);
        CHECK_NEAR(std::abs(normal[2]), 1.0, 1e-10, "EdgeSurf: normal is in z direction");
    }

    // Test 3: No force for separated beam
    {
        Real beam[3] = {1.0, 1.0, 0.5};  // Above shell
        Real forces[3];
        mes.couple_edge_surface(beam, 1, &shell_seg, 1, forces);
        Real mag = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
        CHECK_NEAR(mag, 0.0, 1e-15, "EdgeSurf: zero force for separated beam");
    }

    // Test 4: Nonzero force for penetrating beam
    {
        Real beam[3] = {1.0, 1.0, -0.01};  // Below shell
        Real forces[3];
        Real energy = mes.couple_edge_surface(beam, 1, &shell_seg, 1, forces);
        Real mag = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
        CHECK(mag > 0.0, "EdgeSurf: nonzero force for penetrating beam");
        CHECK(energy > 0.0, "EdgeSurf: positive energy for penetration");
    }

    // Test 5: Force scales with penetration depth
    {
        Real beam1[3] = {1.0, 1.0, -0.01};
        Real beam2[3] = {1.0, 1.0, -0.02};
        Real f1[3], f2[3];
        mes.couple_edge_surface(beam1, 1, &shell_seg, 1, f1);
        mes.couple_edge_surface(beam2, 1, &shell_seg, 1, f2);
        Real mag1 = std::sqrt(f1[0]*f1[0] + f1[1]*f1[1] + f1[2]*f1[2]);
        Real mag2 = std::sqrt(f2[0]*f2[0] + f2[1]*f2[1] + f2[2]*f2[2]);
        CHECK_NEAR(mag2 / mag1, 2.0, 1e-6, "EdgeSurf: force proportional to penetration");
    }

    // Test 6: Energy scales with penetration squared
    {
        Real beam1[3] = {1.0, 1.0, -0.01};
        Real beam2[3] = {1.0, 1.0, -0.02};
        Real f1[3], f2[3];
        Real e1 = mes.couple_edge_surface(beam1, 1, &shell_seg, 1, f1);
        Real e2 = mes.couple_edge_surface(beam2, 1, &shell_seg, 1, f2);
        CHECK_NEAR(e2 / e1, 4.0, 1e-6, "EdgeSurf: energy proportional to penetration^2");
    }

    // Test 7: Force direction is along normal (pushes out)
    {
        Real beam[3] = {1.0, 1.0, -0.01};
        Real forces[3];
        mes.couple_edge_surface(beam, 1, &shell_seg, 1, forces);
        // Force should push beam in +z direction (outward)
        CHECK(forces[2] > 0.0, "EdgeSurf: force pushes beam out of shell");
    }

    // Test 8: Penalty stiffness effect
    {
        MortarEdgeToSurface mes2(2.0e6);
        Real beam[3] = {1.0, 1.0, -0.01};
        Real f1[3], f2[3];
        mes.couple_edge_surface(beam, 1, &shell_seg, 1, f1);
        mes2.couple_edge_surface(beam, 1, &shell_seg, 1, f2);
        Real mag1 = std::sqrt(f1[0]*f1[0] + f1[1]*f1[1] + f1[2]*f1[2]);
        Real mag2 = std::sqrt(f2[0]*f2[0] + f2[1]*f2[1] + f2[2]*f2[2]);
        CHECK_NEAR(mag2 / mag1, 2.0, 1e-6, "EdgeSurf: force scales with penalty stiffness");
    }

    // Test 9: Multiple beam nodes
    {
        Real beams[2 * 3] = {1.0, 1.0, -0.01,  1.0, 1.0, 0.5};
        Real forces[2 * 3];
        mes.couple_edge_surface(beams, 2, &shell_seg, 1, forces);
        // First beam penetrates, second doesn't
        Real mag0 = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
        Real mag1 = std::sqrt(forces[3]*forces[3] + forces[4]*forces[4] + forces[5]*forces[5]);
        CHECK(mag0 > 0.0, "EdgeSurf: penetrating beam node gets force");
        CHECK_NEAR(mag1, 0.0, 1e-15, "EdgeSurf: separated beam node gets no force");
    }
}

// ============================================================================
// 9. MortarAssembly Tests
// ============================================================================
void test_mortar_assembly() {
    std::cout << "\n=== 35c-3: MortarAssembly ===\n";

    MortarAssembly ma;

    // Test 1: Diagonal inversion
    {
        Real D[3] = {2.0, 4.0, 0.5};
        Real D_inv[3];
        ma.invert_diagonal(D, D_inv, 3);
        CHECK_NEAR(D_inv[0], 0.5, 1e-12, "MortarAsm: D_inv[0] = 1/2");
        CHECK_NEAR(D_inv[1], 0.25, 1e-12, "MortarAsm: D_inv[1] = 1/4");
        CHECK_NEAR(D_inv[2], 2.0, 1e-12, "MortarAsm: D_inv[2] = 1/0.5 = 2");
    }

    // Test 2: Zero diagonal element gives zero inverse
    {
        Real D[2] = {1.0, 0.0};
        Real D_inv[2];
        ma.invert_diagonal(D, D_inv, 2);
        CHECK_NEAR(D_inv[1], 0.0, 1e-30, "MortarAsm: zero D gives zero D_inv");
    }

    // Test 3: Coupled stiffness includes master contribution
    {
        Real D[2] = {1.0, 1.0};
        Real M[4] = {0.0, 0.0, 0.0, 0.0};  // Zero coupling
        Real K_m[2] = {100.0, 200.0};
        Real K_s[2] = {50.0, 50.0};
        Real K_c[2];
        ma.assemble_mortar_system(D, M, K_m, K_s, 2, 2, K_c);
        CHECK_NEAR(K_c[0], 100.0, 1e-10, "MortarAsm: zero M gives K_coupled = K_master");
        CHECK_NEAR(K_c[1], 200.0, 1e-10, "MortarAsm: zero M gives K_coupled = K_master [1]");
    }

    // Test 4: Non-zero coupling adds slave contribution
    {
        Real D[1] = {1.0};
        Real M[1] = {0.5};  // Single coupling entry
        Real K_m[1] = {100.0};
        Real K_s[1] = {200.0};
        Real K_c[1];
        ma.assemble_mortar_system(D, M, K_m, K_s, 1, 1, K_c);
        // K_c = 100 + (1.0)^(-1) * 0.5 * 200 * 0.5 * (1.0)^(-1) = 100 + 50 = 150
        CHECK_NEAR(K_c[0], 150.0, 1e-10, "MortarAsm: coupled K = K_m + condensed slave");
    }

    // Test 5: Positivity check
    {
        Real K_pos[3] = {1.0, 2.0, 3.0};
        CHECK(ma.is_positive(K_pos, 3), "MortarAsm: positive K passes check");
    }

    // Test 6: Negative K fails positivity check
    {
        Real K_neg[3] = {1.0, -2.0, 3.0};
        CHECK(!ma.is_positive(K_neg, 3), "MortarAsm: negative K fails positivity check");
    }

    // Test 7: Condition estimate for uniform D
    {
        Real D[4] = {1.0, 1.0, 1.0, 1.0};
        Real cond = ma.condition_estimate(D, 4);
        CHECK_NEAR(cond, 1.0, 1e-12, "MortarAsm: uniform D has condition = 1");
    }

    // Test 8: Condition estimate for non-uniform D
    {
        Real D[3] = {1.0, 10.0, 100.0};
        Real cond = ma.condition_estimate(D, 3);
        CHECK_NEAR(cond, 100.0, 1e-10, "MortarAsm: cond = max/min = 100/1 = 100");
    }

    // Test 9: Assembled K is always >= K_master (slave adds positive contribution)
    {
        Real D[2] = {1.0, 1.0};
        Real M[4] = {0.3, 0.2, 0.1, 0.4};
        Real K_m[2] = {100.0, 200.0};
        Real K_s[2] = {50.0, 80.0};
        Real K_c[2];
        ma.assemble_mortar_system(D, M, K_m, K_s, 2, 2, K_c);
        CHECK(K_c[0] >= K_m[0] - 1e-10, "MortarAsm: K_coupled >= K_master [0]");
        CHECK(K_c[1] >= K_m[1] - 1e-10, "MortarAsm: K_coupled >= K_master [1]");
    }

    // Test 10: Coupled stiffness is positive
    {
        Real D[2] = {1.0, 1.0};
        Real M[4] = {0.3, 0.2, 0.1, 0.4};
        Real K_m[2] = {100.0, 200.0};
        Real K_s[2] = {50.0, 80.0};
        Real K_c[2];
        ma.assemble_mortar_system(D, M, K_m, K_s, 2, 2, K_c);
        CHECK(ma.is_positive(K_c, 2), "MortarAsm: coupled system is positive");
    }
}

// ============================================================================
// 10. MortarThermal Tests
// ============================================================================
void test_mortar_thermal() {
    std::cout << "\n=== 35c-4: MortarThermal ===\n";

    MortarThermal mt;

    // Test 1: Heat flows from hot to cold
    {
        Real T_m[1] = {500.0};
        Real T_s[1] = {300.0};
        Real h_c = 1000.0;
        Real w[1] = {1.0};
        Real flux[1];
        mt.compute_thermal_mortar(T_m, T_s, h_c, w, flux, 1);
        CHECK(flux[0] > 0.0, "MortarTherm: positive flux from hot master to cold slave");
    }

    // Test 2: Flux magnitude
    {
        Real T_m[1] = {500.0};
        Real T_s[1] = {300.0};
        Real h_c = 1000.0;
        Real w[1] = {1.0};
        Real flux[1];
        mt.compute_thermal_mortar(T_m, T_s, h_c, w, flux, 1);
        CHECK_NEAR(flux[0], 200000.0, 1e-6, "MortarTherm: flux = h_c * w * dT = 200000");
    }

    // Test 3: Zero temperature difference gives zero flux
    {
        Real T_m[1] = {400.0};
        Real T_s[1] = {400.0};
        Real h_c = 1000.0;
        Real w[1] = {1.0};
        Real flux[1];
        mt.compute_thermal_mortar(T_m, T_s, h_c, w, flux, 1);
        CHECK_NEAR(flux[0], 0.0, 1e-10, "MortarTherm: zero dT gives zero flux");
    }

    // Test 4: Flux direction reverses
    {
        Real T_m[1] = {300.0};
        Real T_s[1] = {500.0};
        Real h_c = 1000.0;
        Real w[1] = {1.0};
        Real flux[1];
        mt.compute_thermal_mortar(T_m, T_s, h_c, w, flux, 1);
        CHECK(flux[0] < 0.0, "MortarTherm: negative flux from cold master to hot slave");
    }

    // Test 5: Flux scales with h_contact
    {
        Real T_m[1] = {500.0};
        Real T_s[1] = {300.0};
        Real w[1] = {1.0};
        Real flux1[1], flux2[1];
        mt.compute_thermal_mortar(T_m, T_s, 1000.0, w, flux1, 1);
        mt.compute_thermal_mortar(T_m, T_s, 2000.0, w, flux2, 1);
        CHECK_NEAR(flux2[0] / flux1[0], 2.0, 1e-10, "MortarTherm: flux scales with h_c");
    }

    // Test 6: Mortar weight effect
    {
        Real T_m[1] = {500.0};
        Real T_s[1] = {300.0};
        Real w1[1] = {0.5};
        Real w2[1] = {1.0};
        Real flux1[1], flux2[1];
        mt.compute_thermal_mortar(T_m, T_s, 1000.0, w1, flux1, 1);
        mt.compute_thermal_mortar(T_m, T_s, 1000.0, w2, flux2, 1);
        CHECK_NEAR(flux2[0] / flux1[0], 2.0, 1e-10, "MortarTherm: flux scales with mortar weight");
    }

    // Test 7: Interface temperature
    {
        Real T_int = mt.interface_temperature(600.0, 400.0);
        CHECK_NEAR(T_int, 500.0, 1e-12, "MortarTherm: interface temp = (T_m+T_s)/2");
    }

    // Test 8: Thermal dissipation
    {
        Real E = mt.thermal_dissipation(500.0, 300.0, 1000.0, 0.01, 0.001);
        // E = 1000 * 200^2 * 0.01 * 0.001 = 1000 * 40000 * 1e-5 = 400
        CHECK_NEAR(E, 400.0, 1e-6, "MortarTherm: dissipation = h_c * dT^2 * A * dt");
    }

    // Test 9: Thermal conductance
    {
        Real w[3] = {0.5, 1.0, 0.25};
        Real K[3];
        mt.thermal_conductance(1000.0, w, K, 3);
        CHECK_NEAR(K[0], 500.0, 1e-10, "MortarTherm: conductance[0] = h_c * w = 500");
        CHECK_NEAR(K[1], 1000.0, 1e-10, "MortarTherm: conductance[1] = h_c * w = 1000");
        CHECK_NEAR(K[2], 250.0, 1e-10, "MortarTherm: conductance[2] = h_c * w = 250");
    }

    // Test 10: Total heat rate
    {
        Real T_m[3] = {500, 600, 700};
        Real T_s[3] = {300, 300, 300};
        Real w[3] = {1.0, 1.0, 1.0};
        Real flux[3];
        Real total = mt.compute_thermal_mortar(T_m, T_s, 100.0, w, flux, 3);
        // total = 100*(200 + 300 + 400) = 90000
        CHECK_NEAR(total, 90000.0, 1e-6, "MortarTherm: total heat rate = sum of fluxes");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 35 Advanced Contact Algorithms (int22/int24/int25) Test ===\n";

    test_immersed_boundary_contact();
    test_cut_cell_geometry();
    test_immersed_force();
    test_nitsche_contact();
    test_nitsche_shell_solid();
    test_nitsche_pxfem();
    test_mortar_contact_full();
    test_mortar_edge_to_surface();
    test_mortar_assembly();
    test_mortar_thermal();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed) << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
