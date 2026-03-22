/**
 * @file preprocess_wave22_test.cpp
 * @brief Wave 22: Preprocessing Utilities Test Suite (5 components, ~30 tests)
 *
 * Tests 5 components (~6 tests each):
 * - MeshQualityMetrics (hex, tet, shell quality)
 * - MeshRepair (duplicate merge, free edges, T-junctions, renumbering)
 * - AutoContactSurface (exterior faces, grouping, contact pairs)
 * - PartMaterialAssigner (box/sphere region assignment)
 * - CoordinateTransform (cylindrical, spherical, tensor transform)
 */

#include <nexussim/io/preprocess_wave22.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <map>

using namespace nxs;
using namespace nxs::io;

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
// Helper: Unit cube hex element (8 nodes, C-style array)
// ============================================================================
//   Bottom: 0(0,0,0) 1(1,0,0) 2(1,1,0) 3(0,1,0)
//   Top:    4(0,0,1) 5(1,0,1) 6(1,1,1) 7(0,1,1)

static void fill_unit_cube(Real coords[8][3]) {
    Real c[8][3] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            coords[i][j] = c[i][j];
}

static void fill_flat_quad(Real coords[4][3]) {
    Real c[4][3] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}
    };
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            coords[i][j] = c[i][j];
}

// ============================================================================
// 1. MeshQualityMetrics Tests
// ============================================================================
void test_mesh_quality_metrics() {
    std::cout << "\n=== MeshQualityMetrics Tests ===\n";

    // Test 1: Unit cube hex -- ideal element: aspect ratio = 1.0, skewness = 0
    {
        Real coords[8][3];
        fill_unit_cube(coords);
        auto q = MeshQualityMetrics::compute_hex_quality(coords);
        CHECK_NEAR(q.aspect_ratio, 1.0, 1e-10,
                   "hex unit cube: aspect ratio = 1.0");
        CHECK_NEAR(q.jacobian_ratio, 1.0, 1e-10,
                   "hex unit cube: Jacobian ratio = 1.0");
        CHECK_NEAR(q.skewness, 0.0, 1e-10,
                   "hex unit cube: skewness = 0.0");
        CHECK_NEAR(q.warpage, 0.0, 1e-10,
                   "hex unit cube: warpage = 0.0");
    }

    // Test 2: Regular tet quality
    {
        Real tet[4][3] = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.5, std::sqrt(3.0)/2.0, 0.0},
            {0.5, std::sqrt(3.0)/6.0, std::sqrt(6.0)/3.0}
        };
        auto q = MeshQualityMetrics::compute_tet_quality(tet);
        CHECK_NEAR(q.aspect_ratio, 1.0, 0.01,
                   "regular tet: aspect ratio ~ 1.0");
        CHECK(q.is_valid, "regular tet: is valid");
    }

    // Test 3: Shell quality on flat quad -- warpage = 0, all angles 90 deg
    {
        Real quad[4][3];
        fill_flat_quad(quad);
        auto q = MeshQualityMetrics::compute_shell_quality(quad);
        CHECK_NEAR(q.warpage, 0.0, 1e-10,
                   "flat quad: warpage = 0 degrees");
        CHECK_NEAR(q.aspect_ratio, 1.0, 1e-10,
                   "flat quad: aspect ratio = 1.0");
        CHECK_NEAR(q.min_angle, 90.0, 1e-6,
                   "flat quad: min angle = 90 deg");
        CHECK_NEAR(q.max_angle, 90.0, 1e-6,
                   "flat quad: max angle = 90 deg");
    }

    // Test 4: Distorted hex (elongated 10:1 in x-direction)
    {
        Real coords[8][3] = {
            {0,0,0}, {10,0,0}, {10,1,0}, {0,1,0},
            {0,0,1}, {10,0,1}, {10,1,1}, {0,1,1}
        };
        auto q = MeshQualityMetrics::compute_hex_quality(coords);
        CHECK_NEAR(q.aspect_ratio, 10.0, 1e-10,
                   "elongated hex: aspect ratio = 10.0");
        CHECK(q.aspect_ratio > 5.0,
              "elongated hex: aspect ratio > 5.0 (distorted)");
    }

    // Test 5: Warped shell (one node lifted out of plane)
    {
        Real warped[4][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0.5}
        };
        auto q = MeshQualityMetrics::compute_shell_quality(warped);
        CHECK(q.warpage > 1.0,
              "warped quad: non-zero warpage (> 1 degree)");
        CHECK(q.is_valid, "warped quad: still valid");
    }

    // Test 6: Inverted tet (negative volume) is marked invalid
    {
        Real inv_tet[4][3] = {
            {0,0,0}, {1,0,0}, {0,1,0}, {0,0,-1}
        };
        auto q = MeshQualityMetrics::compute_tet_quality(inv_tet);
        CHECK(!q.is_valid, "inverted tet: marked invalid");
    }
}

// ============================================================================
// 2. MeshRepair Tests
// ============================================================================
void test_mesh_repair() {
    std::cout << "\n=== MeshRepair Tests ===\n";

    // Test 1: Detect and merge duplicate nodes
    {
        std::vector<MeshRepair::Node> nodes = {
            {1, 0.0, 0.0, 0.0},
            {2, 1.0, 0.0, 0.0},
            {3, 0.0, 0.0, 1e-12},  // near-duplicate of node 1
            {4, 2.0, 0.0, 0.0}
        };
        auto result = MeshRepair::merge_duplicate_nodes(nodes, 1e-10);
        CHECK(result.num_merged == 1, "merge: 1 duplicate found");
        CHECK(nodes.size() == 3, "merge: 3 nodes remain after merging");
        CHECK(result.merge_map.size() == 1, "merge: merge_map has 1 entry");
        CHECK(result.merge_map[0].first == 3 && result.merge_map[0].second == 1,
              "merge: node 3 merged into node 1");
    }

    // Test 2: No duplicates to merge -- all nodes preserved
    {
        std::vector<MeshRepair::Node> nodes = {
            {1, 0.0, 0.0, 0.0},
            {2, 1.0, 0.0, 0.0},
            {3, 0.0, 1.0, 0.0},
            {4, 0.0, 0.0, 1.0}
        };
        auto result = MeshRepair::merge_duplicate_nodes(nodes, 1e-10);
        CHECK(result.num_merged == 0, "no duplicates: num_merged = 0");
        CHECK(nodes.size() == 4, "no duplicates: all 4 nodes remain");
    }

    // Test 3: Free edge detection on single quad (4 free edges)
    {
        std::vector<MeshRepair::Element> elements = {
            {1, {1, 2, 3, 4}}
        };
        auto free = MeshRepair::find_free_edges(elements);
        CHECK(free.size() == 4, "single quad: 4 free edges");
    }

    // Test 4: Closed cube surface has 0 free edges
    {
        // 6 quad faces of a cube: each edge shared by exactly 2 faces
        std::vector<MeshRepair::Element> elements = {
            {1, {1,2,3,4}},   // bottom
            {2, {5,6,7,8}},   // top
            {3, {1,2,6,5}},   // front
            {4, {4,3,7,8}},   // back
            {5, {1,4,8,5}},   // left
            {6, {2,3,7,6}}    // right
        };
        auto free = MeshRepair::find_free_edges(elements);
        CHECK(free.size() == 0, "closed cube surface: 0 free edges");
    }

    // Test 5: T-junction detection (edge shared by 3 elements)
    {
        std::vector<MeshRepair::Element> elements = {
            {1, {1, 2, 3, 4}},
            {2, {2, 5, 6, 3}},
            {3, {2, 7, 8, 3}}  // edge 2-3 shared by 3 elements
        };
        auto tjunc = MeshRepair::find_t_junctions(elements);
        CHECK(tjunc.size() >= 1, "T-junction: at least 1 T-junction edge detected");
    }

    // Test 6: Apply merge map to element connectivity
    {
        std::vector<MeshRepair::Element> elements = {
            {1, {1, 3, 4}}  // node 3 was merged into node 1
        };
        std::vector<std::pair<int, int>> merge_map = {{3, 1}};
        MeshRepair::apply_merge_to_elements(elements, merge_map);
        CHECK(elements[0].node_ids[1] == 1,
              "apply_merge: node 3 replaced with node 1 in element");
    }
}

// ============================================================================
// 3. AutoContactSurface Tests
// ============================================================================
void test_auto_contact_surface() {
    std::cout << "\n=== AutoContactSurface Tests ===\n";

    // Setup: unit cube hex with SoA node coords
    // Node IDs: 1..8, array indices: 0..7
    Real node_x[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    Real node_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    Real node_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    std::map<int, int> nid_to_idx;
    for (int i = 0; i < 8; i++) nid_to_idx[i + 1] = i;

    // Test 1: Single hex has 6 exterior faces
    {
        int conn[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int parts[1] = {1};
        auto faces = AutoContactSurface::extract_exterior_faces(
            conn, 8, 1, parts, node_x, node_y, node_z, nid_to_idx);
        CHECK(faces.size() == 6, "single hex: 6 exterior faces");
    }

    // Test 2: All exterior face normals are unit vectors
    {
        int conn[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int parts[1] = {1};
        auto faces = AutoContactSurface::extract_exterior_faces(
            conn, 8, 1, parts, node_x, node_y, node_z, nid_to_idx);
        bool all_unit = true;
        for (const auto& f : faces) {
            Real mag = std::sqrt(f.normal[0]*f.normal[0] +
                                 f.normal[1]*f.normal[1] +
                                 f.normal[2]*f.normal[2]);
            if (std::abs(mag - 1.0) > 0.01) all_unit = false;
        }
        CHECK(all_unit, "single hex: all face normals are unit vectors");
    }

    // Test 3: Group by part returns the correct part
    {
        int conn[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int parts[1] = {42};
        auto faces = AutoContactSurface::extract_exterior_faces(
            conn, 8, 1, parts, node_x, node_y, node_z, nid_to_idx);
        auto grouped = AutoContactSurface::group_by_part(faces);
        CHECK(grouped.count(42) > 0, "group_by_part: part 42 found");
        CHECK(grouped[42].size() == 6, "group_by_part: part 42 has 6 faces");
    }

    // Test 4: Contact pair creation from 2 parts
    {
        std::map<int, std::vector<ExteriorFace>> faces_by_part;
        ExteriorFace dummy{};
        faces_by_part[1].push_back(dummy);
        faces_by_part[2].push_back(dummy);

        auto pairs = AutoContactSurface::generate_all_pairs(faces_by_part);
        CHECK(pairs.size() == 1, "2 parts: 1 contact pair");
        CHECK(pairs[0].part1 == 1 && pairs[0].part2 == 2,
              "contact pair: (part1=1, part2=2)");
    }

    // Test 5: Contact pairs from 3 parts => C(3,2) = 3 pairs
    {
        std::map<int, std::vector<ExteriorFace>> faces_by_part;
        ExteriorFace dummy{};
        faces_by_part[1].push_back(dummy);
        faces_by_part[2].push_back(dummy);
        faces_by_part[3].push_back(dummy);

        auto pairs = AutoContactSurface::generate_all_pairs(faces_by_part);
        CHECK(pairs.size() == 3, "3 parts: 3 contact pairs");
    }

    // Test 6: Two adjacent hexes sharing a face -> 10 exterior faces
    {
        // Second hex shares face: nodes 2,3,6,7 (IDs 2,3,7,6)
        // New nodes: 9(2,0,0), 10(2,1,0), 11(2,0,1), 12(2,1,1)
        Real nx[12] = {0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2};
        Real ny[12] = {0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1};
        Real nz[12] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1};

        std::map<int, int> nmap;
        for (int i = 0; i < 12; i++) nmap[i + 1] = i;

        // Two hex elements (connectivity is 8 nodes each, laid out contiguously)
        int conn[16] = {
            1, 2, 3, 4, 5, 6, 7, 8,        // hex 1
            2, 9, 10, 3, 6, 11, 12, 7       // hex 2
        };
        int parts[2] = {1, 1};

        auto faces = AutoContactSurface::extract_exterior_faces(
            conn, 8, 2, parts, nx, ny, nz, nmap);
        // 2 hexes = 12 total faces; 1 shared internal face counted twice -> 2 internal
        // Exterior faces: 12 - 2 = 10
        CHECK(faces.size() == 10, "2 adjacent hexes: 10 exterior faces");
    }
}

// ============================================================================
// 4. PartMaterialAssigner Tests
// ============================================================================
void test_part_material_assigner() {
    std::cout << "\n=== PartMaterialAssigner Tests ===\n";

    // Unit cube: 8 nodes, centroid at (0.5, 0.5, 0.5)
    Real node_x[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    Real node_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    Real node_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    std::map<int, int> nmap;
    for (int i = 0; i < 8; i++) nmap[i + 1] = i;

    std::vector<int> elem_ids = {1};
    std::vector<std::vector<int>> elem_node_ids = {{1, 2, 3, 4, 5, 6, 7, 8}};

    // Test 1: Box region fully containing the element
    {
        Real box_min[3] = {-1, -1, -1};
        Real box_max[3] = {2, 2, 2};
        auto a = PartMaterialAssigner::assign_by_box(
            elem_ids, elem_node_ids, node_x, node_y, node_z,
            nmap, box_min, box_max, 10);
        CHECK(a.size() == 1, "box region: 1 element assigned");
        CHECK(a[0].part_id == 10, "box region: part_id = 10");
    }

    // Test 2: Box region not containing the element
    {
        Real box_min[3] = {5, 5, 5};
        Real box_max[3] = {10, 10, 10};
        auto a = PartMaterialAssigner::assign_by_box(
            elem_ids, elem_node_ids, node_x, node_y, node_z,
            nmap, box_min, box_max, 20);
        CHECK(a.size() == 0, "far box: no elements assigned");
    }

    // Test 3: Sphere region containing the element centroid
    {
        Real center[3] = {0.5, 0.5, 0.5};
        auto a = PartMaterialAssigner::assign_by_sphere(
            elem_ids, elem_node_ids, node_x, node_y, node_z,
            nmap, center, 1.0, 30);
        CHECK(a.size() == 1, "sphere region: 1 element assigned");
        CHECK(a[0].part_id == 30, "sphere region: part_id = 30");
    }

    // Test 4: Sphere region too far to contain centroid
    {
        Real center[3] = {10, 10, 10};
        auto a = PartMaterialAssigner::assign_by_sphere(
            elem_ids, elem_node_ids, node_x, node_y, node_z,
            nmap, center, 0.1, 40);
        CHECK(a.size() == 0, "far sphere: no elements assigned");
    }

    // Test 5: assign_material_to_part returns correct mapping
    {
        auto p = PartMaterialAssigner::assign_material_to_part(5, 100);
        CHECK(p.first == 5 && p.second == 100,
              "assign_material_to_part: (5, 100)");
    }

    // Test 6: Cylinder region containing the element centroid
    {
        Real origin[3] = {0.5, 0.5, -1.0};
        Real axis[3]   = {0.0, 0.0, 1.0};
        auto a = PartMaterialAssigner::assign_by_cylinder(
            elem_ids, elem_node_ids, node_x, node_y, node_z,
            nmap, origin, axis, 1.0, 5.0, 50);
        CHECK(a.size() == 1, "cylinder region: 1 element assigned");
        CHECK(a[0].part_id == 50, "cylinder region: part_id = 50");
    }
}

// ============================================================================
// 5. CoordinateTransform Tests
// ============================================================================
void test_coordinate_transform() {
    std::cout << "\n=== CoordinateTransform Tests ===\n";

    // Test 1: Cartesian to cylindrical (point on x-axis, z-aligned axis)
    {
        Real x[1] = {3.0}, y[1] = {0.0}, z[1] = {5.0};
        Real origin[3] = {0, 0, 0};
        Real axis[3] = {0, 0, 1};
        CoordinateTransform::to_cylindrical(x, y, z, 1, origin, axis);
        CHECK_NEAR(x[0], 3.0, 1e-10, "cyl: (3,0,5) -> r=3");
        CHECK_NEAR(y[0], 0.0, 1e-10, "cyl: (3,0,5) -> theta=0");
        CHECK_NEAR(z[0], 5.0, 1e-10, "cyl: (3,0,5) -> z=5");
    }

    // Test 2: Cartesian to cylindrical (point at 45 degrees in xy-plane)
    {
        Real v = 1.0 / std::sqrt(2.0);
        Real x[1] = {v}, y[1] = {v}, z[1] = {0.0};
        Real origin[3] = {0, 0, 0};
        Real axis[3] = {0, 0, 1};
        CoordinateTransform::to_cylindrical(x, y, z, 1, origin, axis);
        CHECK_NEAR(x[0], 1.0, 1e-10, "cyl: (v,v,0) -> r=1.0");
        CHECK_NEAR(y[0], M_PI / 4.0, 1e-10, "cyl: (v,v,0) -> theta=pi/4");
    }

    // Test 3: Cylindrical round-trip (to_cylindrical then from_cylindrical)
    {
        Real x[1] = {2.0}, y[1] = {3.0}, z[1] = {7.0};
        Real origin[3] = {0, 0, 0};
        Real axis[3] = {0, 0, 1};
        CoordinateTransform::to_cylindrical(x, y, z, 1, origin, axis);
        CoordinateTransform::from_cylindrical(x, y, z, 1, origin, axis);
        CHECK_NEAR(x[0], 2.0, 1e-10, "cyl round-trip: x=2.0");
        CHECK_NEAR(y[0], 3.0, 1e-10, "cyl round-trip: y=3.0");
        CHECK_NEAR(z[0], 7.0, 1e-10, "cyl round-trip: z=7.0");
    }

    // Test 4: Cartesian to spherical (point on z-axis: north pole)
    {
        Real x[1] = {0.0}, y[1] = {0.0}, z[1] = {5.0};
        Real origin[3] = {0, 0, 0};
        CoordinateTransform::to_spherical(x, y, z, 1, origin);
        CHECK_NEAR(x[0], 5.0, 1e-10, "sph: (0,0,5) -> r=5");
        CHECK_NEAR(y[0], 0.0, 1e-10, "sph: (0,0,5) -> theta=0 (north pole)");
    }

    // Test 5: Spherical round-trip (to_spherical then from_spherical)
    {
        Real x[1] = {1.0}, y[1] = {2.0}, z[1] = {3.0};
        Real origin[3] = {0, 0, 0};
        CoordinateTransform::to_spherical(x, y, z, 1, origin);
        CoordinateTransform::from_spherical(x, y, z, 1, origin);
        CHECK_NEAR(x[0], 1.0, 1e-10, "sph round-trip: x=1.0");
        CHECK_NEAR(y[0], 2.0, 1e-10, "sph round-trip: y=2.0");
        CHECK_NEAR(z[0], 3.0, 1e-10, "sph round-trip: z=3.0");
    }

    // Test 6: Tensor transformation with identity preserves tensor
    {
        Real tensor[6] = {100, 200, 300, 10, 20, 30};
        Real I[9] = {1,0,0, 0,1,0, 0,0,1};
        Real result[6];
        CoordinateTransform::transform_tensor(tensor, I, result);
        CHECK_NEAR(result[0], 100.0, 1e-10, "tensor identity: Txx preserved");
        CHECK_NEAR(result[1], 200.0, 1e-10, "tensor identity: Tyy preserved");
        CHECK_NEAR(result[2], 300.0, 1e-10, "tensor identity: Tzz preserved");
        CHECK_NEAR(result[3], 10.0, 1e-10, "tensor identity: Txy preserved");
    }

    // Test 7: Tensor trace (I1) is invariant under 90-deg z-rotation
    {
        Real tensor[6] = {100, 200, 300, 10, 20, 30};
        Real trace_orig = tensor[0] + tensor[1] + tensor[2];

        // 90-degree rotation about z: Q = [[0,-1,0],[1,0,0],[0,0,1]]
        Real Rz[9] = {0,-1,0, 1,0,0, 0,0,1};
        Real result[6];
        CoordinateTransform::transform_tensor(tensor, Rz, result);
        Real trace_rot = result[0] + result[1] + result[2];
        CHECK_NEAR(trace_rot, trace_orig, 1e-10,
                   "tensor trace invariant under 90-deg z-rotation");
    }

    // Test 8: J2 (second deviatoric invariant) is invariant under rotation
    {
        Real tensor[6] = {100, 200, 300, 10, 20, 30};
        // Compute J2 for original
        Real p = (tensor[0] + tensor[1] + tensor[2]) / 3.0;
        Real s0 = tensor[0] - p, s1 = tensor[1] - p, s2 = tensor[2] - p;
        Real J2_orig = 0.5*(s0*s0 + s1*s1 + s2*s2) +
                       tensor[3]*tensor[3] + tensor[4]*tensor[4] + tensor[5]*tensor[5];

        Real Rz[9] = {0,-1,0, 1,0,0, 0,0,1};
        Real result[6];
        CoordinateTransform::transform_tensor(tensor, Rz, result);

        Real pr = (result[0] + result[1] + result[2]) / 3.0;
        Real sr0 = result[0]-pr, sr1 = result[1]-pr, sr2 = result[2]-pr;
        Real J2_rot = 0.5*(sr0*sr0 + sr1*sr1 + sr2*sr2) +
                      result[3]*result[3] + result[4]*result[4] + result[5]*result[5];

        CHECK_NEAR(J2_rot, J2_orig, 1e-8,
                   "tensor J2 invariant under rotation");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "===================================================\n";
    std::cout << "Wave 22: Preprocessing Utilities Test Suite\n";
    std::cout << "===================================================\n";

    test_mesh_quality_metrics();
    test_mesh_repair();
    test_auto_contact_surface();
    test_part_material_assigner();
    test_coordinate_transform();

    std::cout << "\n===================================\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";

    if (tests_failed > 0) {
        std::cout << "RESULT: FAIL\n";
        return 1;
    }
    std::cout << "RESULT: PASS\n";
    return 0;
}
