/**
 * @file contact_wave43_test.cpp
 * @brief Wave 43: Spatial bucket sort contact search — ~60 tests
 */

#include <nexussim/fem/contact_wave43.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>

// ============================================================================
// Test harness
// ============================================================================

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

using namespace nxs::fem;

// ============================================================================
// Helper — build a simple AABB from two corner arrays
// ============================================================================
static AABB make_aabb(double x0, double y0, double z0,
                      double x1, double y1, double z1)
{
    Real lo[3] = {Real(x0), Real(y0), Real(z0)};
    Real hi[3] = {Real(x1), Real(y1), Real(z1)};
    return AABB(lo, hi);
}

// ============================================================================
// Test groups
// ============================================================================

// --- AABB basic construction -------------------------------------------------
static void test_aabb_construction()
{
    // Default constructor gives zero box
    AABB a;
    CHECK(a.is_valid(), "AABB default: valid");
    CHECK_NEAR(a.min_pt[0], 0.0, 1e-14, "AABB default: min x");
    CHECK_NEAR(a.max_pt[0], 0.0, 1e-14, "AABB default: max x");

    // Explicit construction
    AABB b = make_aabb(1, 2, 3, 4, 5, 6);
    CHECK(b.is_valid(), "AABB explicit: valid");
    CHECK_NEAR(b.min_pt[1], 2.0, 1e-14, "AABB explicit: min y");
    CHECK_NEAR(b.max_pt[2], 6.0, 1e-14, "AABB explicit: max z");

    // empty() gives inside-out box
    AABB e = AABB::empty();
    CHECK(!e.is_valid(), "AABB::empty: not valid (inside-out)");
}

// --- AABB expand with point --------------------------------------------------
static void test_aabb_expand_point()
{
    AABB b = AABB::empty();
    Real p1[3] = {1, 2, 3};
    Real p2[3] = {4, 5, 6};
    Real p3[3] = {-1, 3, 7};
    b.expand(p1);
    b.expand(p2);
    b.expand(p3);

    CHECK_NEAR(b.min_pt[0], -1.0, 1e-14, "expand_point: min x");
    CHECK_NEAR(b.min_pt[1],  2.0, 1e-14, "expand_point: min y");
    CHECK_NEAR(b.min_pt[2],  3.0, 1e-14, "expand_point: min z");
    CHECK_NEAR(b.max_pt[0],  4.0, 1e-14, "expand_point: max x");
    CHECK_NEAR(b.max_pt[1],  5.0, 1e-14, "expand_point: max y");
    CHECK_NEAR(b.max_pt[2],  7.0, 1e-14, "expand_point: max z");
    CHECK(b.is_valid(), "expand_point: valid after expand");
}

// --- AABB expand with tolerance ---------------------------------------------
static void test_aabb_expand_tol()
{
    AABB b = make_aabb(0, 0, 0, 1, 1, 1);
    b.expand(Real(0.5));
    CHECK_NEAR(b.min_pt[0], -0.5, 1e-14, "expand_tol: min x");
    CHECK_NEAR(b.max_pt[0],  1.5, 1e-14, "expand_tol: max x");
    CHECK_NEAR(b.min_pt[2], -0.5, 1e-14, "expand_tol: min z");
    CHECK_NEAR(b.max_pt[2],  1.5, 1e-14, "expand_tol: max z");
}

// --- AABB overlaps -----------------------------------------------------------
static void test_aabb_overlaps()
{
    AABB a = make_aabb(0, 0, 0, 1, 1, 1);
    AABB b = make_aabb(0.5, 0.5, 0.5, 2, 2, 2); // overlaps a
    AABB c = make_aabb(2, 2, 2, 3, 3, 3);        // separated
    AABB d = make_aabb(1, 0, 0, 2, 1, 1);        // touching at x=1 face

    CHECK( a.overlaps(b), "overlaps: partial overlap");
    CHECK(!a.overlaps(c), "overlaps: separated");
    CHECK( a.overlaps(d), "overlaps: touching face");
    CHECK( b.overlaps(a), "overlaps: symmetry");
    CHECK(!c.overlaps(a), "overlaps: separated symmetry");
    CHECK( a.overlaps(a), "overlaps: self");
}

// --- AABB merge -------------------------------------------------------------
static void test_aabb_merge()
{
    AABB a = make_aabb(0, 0, 0, 1, 1, 1);
    AABB b = make_aabb(2, 2, 2, 3, 3, 3);
    AABB m = a.merge(b);

    CHECK_NEAR(m.min_pt[0], 0.0, 1e-14, "merge: min x");
    CHECK_NEAR(m.min_pt[1], 0.0, 1e-14, "merge: min y");
    CHECK_NEAR(m.max_pt[0], 3.0, 1e-14, "merge: max x");
    CHECK_NEAR(m.max_pt[2], 3.0, 1e-14, "merge: max z");
    CHECK(m.is_valid(), "merge: valid");
}

// --- AABB centre & half-extents ---------------------------------------------
static void test_aabb_centre()
{
    AABB b = make_aabb(0, 0, 0, 2, 4, 6);
    Real c[3], h[3];
    b.centre(c);
    b.half_extents(h);

    CHECK_NEAR(c[0], 1.0, 1e-14, "centre: x");
    CHECK_NEAR(c[1], 2.0, 1e-14, "centre: y");
    CHECK_NEAR(c[2], 3.0, 1e-14, "centre: z");
    CHECK_NEAR(h[0], 1.0, 1e-14, "half_extents: x");
    CHECK_NEAR(h[1], 2.0, 1e-14, "half_extents: y");
    CHECK_NEAR(h[2], 3.0, 1e-14, "half_extents: z");
}

// --- AABB surface area -------------------------------------------------------
static void test_aabb_surface_area()
{
    AABB b = make_aabb(0, 0, 0, 1, 2, 3);
    // SA = 2*(1*2 + 2*3 + 3*1) = 2*(2+6+3) = 22
    CHECK_NEAR(b.surface_area(), 22.0, 1e-12, "surface_area: unit box");

    AABB unit = make_aabb(0,0,0,1,1,1);
    CHECK_NEAR(unit.surface_area(), 6.0, 1e-12, "surface_area: 1x1x1");
}

// --- BucketSort3D basic insert/query ----------------------------------------
static void test_bucket_insert_query_basic()
{
    AABB domain = make_aabb(0,0,0, 10,10,10);
    BucketSort3D grid(domain, 2.0);

    // Grid should be 5x5x5 = 125 buckets
    int dims[3];
    grid.grid_dims(dims);
    CHECK(dims[0] == 5 && dims[1] == 5 && dims[2] == 5,
          "bucket3d: 5x5x5 dims");
    CHECK(grid.num_buckets() == 125, "bucket3d: 125 total buckets");

    // Insert two items
    AABB box1 = make_aabb(0.5, 0.5, 0.5, 1.5, 1.5, 1.5); // centre (1,1,1) -> bucket (0,0,0)
    AABB box2 = make_aabb(5.5, 5.5, 5.5, 6.5, 6.5, 6.5); // centre (6,6,6) -> bucket (3,3,3)
    grid.insert(0, box1);
    grid.insert(1, box2);

    CHECK(grid.total_entries() == 2, "bucket3d: 2 entries after insert");

    // Query near box1
    std::vector<int> found;
    grid.query(make_aabb(0,0,0, 2,2,2), [&](int id){ found.push_back(id); });
    CHECK(std::find(found.begin(), found.end(), 0) != found.end(),
          "bucket3d: query finds box1");
    CHECK(std::find(found.begin(), found.end(), 1) == found.end(),
          "bucket3d: query does not find far box2");
}

// --- BucketSort3D clear ------------------------------------------------------
static void test_bucket_clear()
{
    AABB domain = make_aabb(0,0,0, 10,10,10);
    BucketSort3D grid(domain, 2.0);
    AABB b = make_aabb(1,1,1, 2,2,2);
    grid.insert(42, b);
    CHECK(grid.total_entries() == 1, "bucket clear: 1 before clear");
    grid.clear();
    CHECK(grid.total_entries() == 0, "bucket clear: 0 after clear");
}

// --- BucketSort3D out-of-domain clamping ------------------------------------
static void test_bucket_clamping()
{
    AABB domain = make_aabb(0,0,0, 5,5,5);
    BucketSort3D grid(domain, 1.0);

    // Insert a box whose centre is outside the domain
    AABB oob = make_aabb(7, 7, 7, 8, 8, 8); // centre (7.5,7.5,7.5) — clamped
    grid.insert(99, oob);
    CHECK(grid.total_entries() == 1, "bucket_clamping: entry inserted despite OOB");

    // Query entire domain — should find it (clamped to corner bucket)
    std::vector<int> found;
    grid.query(make_aabb(0,0,0, 6,6,6), [&](int id){ found.push_back(id); });
    CHECK(std::find(found.begin(), found.end(), 99) != found.end(),
          "bucket_clamping: clamped entry found by wide query");
}

// --- BucketSort3D: multi-bucket query range ---------------------------------
static void test_bucket_multi_range_query()
{
    AABB domain = make_aabb(0,0,0, 10,10,10);
    BucketSort3D grid(domain, 1.0);

    // Insert 8 boxes at corners of a 2x2x2 region
    int id = 0;
    for (double x : {0.5, 1.5}) for (double y : {0.5, 1.5}) for (double z : {0.5, 1.5}) {
        AABB b = make_aabb(x-0.4, y-0.4, z-0.4, x+0.4, y+0.4, z+0.4);
        grid.insert(id++, b);
    }
    CHECK(grid.total_entries() == 8, "multi_range: 8 entries");

    // A query box spanning [0,2]^3 should find all 8
    std::unordered_set<int> found;
    grid.query(make_aabb(0,0,0, 2,2,2), [&](int i){ found.insert(i); });
    CHECK(static_cast<int>(found.size()) == 8, "multi_range: all 8 found");

    // A query box spanning [0, 0.9]^3 stays within bucket 0 on each axis
    // (floor(0.9/1.0)=0), so only the 4 segments with centres at (0.5,*,*) etc. are found.
    found.clear();
    grid.query(make_aabb(0,0,0, 0.9,0.9,0.9), [&](int i){ found.insert(i); });
    // Only the single bucket (0,0,0) is queried; it holds the segment with centre (0.5,0.5,0.5)
    CHECK(static_cast<int>(found.size()) == 1, "multi_range: 1 in strict first-octant bucket");
}

// --- ContactBucketSearch: simple quad mesh -----------------------------------
static void test_contact_search_quads()
{
    // Two coplanar quad surfaces facing each other, slightly separated.
    // Surface A: 4 quads in z=0 plane (y=0..2, x=0..2)
    // Surface B: 4 quads in z=0.1 plane (should generate contact candidates)

    // 9 nodes per surface arranged in 3x3 grid
    auto node = [](int i, int j, double z) -> std::array<double,3> {
        return {static_cast<double>(i), static_cast<double>(j), z};
    };

    // Surface A nodes (z=0): indices 0..8
    // Surface B nodes (z=0.1): indices 9..17
    std::vector<Real> pos;
    for (int j = 0; j <= 2; ++j) for (int i = 0; i <= 2; ++i) {
        auto n = node(i, j, 0.0);
        pos.push_back(Real(n[0])); pos.push_back(Real(n[1])); pos.push_back(Real(n[2]));
    }
    for (int j = 0; j <= 2; ++j) for (int i = 0; i <= 2; ++i) {
        auto n = node(i, j, 0.1);
        pos.push_back(Real(n[0])); pos.push_back(Real(n[1])); pos.push_back(Real(n[2]));
    }

    // Quad connectivity for a 2x2 mesh (4 quads per surface)
    // Node numbering in row-major: node(i,j) = j*3 + i
    auto nid = [](int i, int j) { return j*3 + i; };
    std::vector<int> conn;
    for (int jq = 0; jq < 2; ++jq) for (int iq = 0; iq < 2; ++iq) {
        conn.push_back(nid(iq,  jq));
        conn.push_back(nid(iq+1,jq));
        conn.push_back(nid(iq+1,jq+1));
        conn.push_back(nid(iq,  jq+1));
    }
    // Surface B offsets every node by 9
    for (int jq = 0; jq < 2; ++jq) for (int iq = 0; iq < 2; ++iq) {
        conn.push_back(9 + nid(iq,  jq));
        conn.push_back(9 + nid(iq+1,jq));
        conn.push_back(9 + nid(iq+1,jq+1));
        conn.push_back(9 + nid(iq,  jq+1));
    }

    ContactBucketSearch searcher;
    searcher.build(pos.data(), 18, conn.data(), 8, 4, /*padding=*/Real(0));

    CHECK(searcher.num_segments() == 8, "quad search: 8 segments built");
    CHECK(searcher.domain().is_valid(), "quad search: domain valid");

    // With gap_tolerance = 0.2, all A-B pairs should be candidates
    auto pairs = searcher.find_pairs(Real(0.2));
    CHECK(!pairs.empty(), "quad search: non-empty pair list with tolerance");

    // At least some pairs should cross surfaces (A seg 0..3 paired with B seg 4..7).
    // Same-surface pairs also appear because adjacent quads share edges and their
    // AABBs touch, so we only assert at least one cross-surface pair exists.
    bool any_cross = false;
    for (const auto& p : pairs) {
        bool s1_A = p.seg1 < 4;
        bool s2_B = p.seg2 >= 4;
        bool s1_B = p.seg1 >= 4;
        bool s2_A = p.seg2 < 4;
        if ((s1_A && s2_B) || (s1_B && s2_A)) { any_cross = true; break; }
    }
    CHECK(any_cross, "quad search: at least one cross-surface pair found");
}

// --- ContactBucketSearch: no false negatives --------------------------------
static void test_contact_no_false_negatives()
{
    // Two boxes that clearly overlap — must produce at least one pair
    std::vector<AABB> aabbs;
    aabbs.push_back(make_aabb(0, 0, 0, 1, 1, 1));
    aabbs.push_back(make_aabb(0.5, 0.5, 0.5, 1.5, 1.5, 1.5));

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));
    CHECK(!pairs.empty(), "no_false_neg: overlapping boxes produce a pair");
}

// --- ContactBucketSearch: far-apart segments — no false positives -----------
static void test_contact_far_apart()
{
    std::vector<AABB> aabbs;
    aabbs.push_back(make_aabb(0, 0, 0, 1, 1, 1));
    aabbs.push_back(make_aabb(100, 100, 100, 101, 101, 101));

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));
    CHECK(pairs.empty(), "far_apart: no pairs for separated segments");
}

// --- ContactBucketSearch: coincident segments --------------------------------
static void test_contact_coincident()
{
    AABB b = make_aabb(0, 0, 0, 1, 1, 1);
    std::vector<AABB> aabbs = {b, b, b}; // three identical boxes

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));
    // Should find 3 pairs: (0,1),(0,2),(1,2)
    CHECK(static_cast<int>(pairs.size()) == 3, "coincident: 3 pairs for 3 identical boxes");
}

// --- ContactBucketSearch: touching (gap == 0) --------------------------------
static void test_contact_touching()
{
    // Two boxes sharing an exact face
    std::vector<AABB> aabbs;
    aabbs.push_back(make_aabb(0, 0, 0, 1, 1, 1));
    aabbs.push_back(make_aabb(1, 0, 0, 2, 1, 1));

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));
    CHECK(!pairs.empty(), "touching: touching boxes produce a pair");
    if (!pairs.empty()) {
        CHECK(pairs[0].gap_estimate <= Real(1e-10),
              "touching: gap_estimate near zero for touching faces");
    }
}

// --- ContactBucketSearch: gap estimate sign convention ----------------------
static void test_contact_gap_sign()
{
    // Overlapping: gap should be <= 0
    std::vector<AABB> ov;
    ov.push_back(make_aabb(0, 0, 0, 2, 2, 2));
    ov.push_back(make_aabb(1, 1, 1, 3, 3, 3));
    ContactBucketSearch s1; s1.build(ov);
    auto p1 = s1.find_pairs();
    CHECK(!p1.empty() && p1[0].gap_estimate <= Real(0),
          "gap_sign: overlapping => gap <= 0");

    // Separated: gap should be > 0 (but within tolerance won't appear unless pad is used)
    std::vector<AABB> sep;
    sep.push_back(make_aabb(0, 0, 0, 1, 1, 1));
    sep.push_back(make_aabb(1.5, 0, 0, 2.5, 1, 1));
    ContactBucketSearch s2; s2.build(sep);
    auto p2 = s2.find_pairs(Real(1.0)); // tolerance large enough to find it
    CHECK(!p2.empty() && p2[0].gap_estimate > Real(0),
          "gap_sign: separated + tolerance => gap > 0");
}

// --- Scaling: 100-element grid -----------------------------------------------
static void test_scaling_100()
{
    // Build a 10x10 flat quad surface (100 quads, 121 nodes)
    const int N = 10;
    std::vector<Real> pos;
    for (int j = 0; j <= N; ++j) for (int i = 0; i <= N; ++i) {
        pos.push_back(Real(i));
        pos.push_back(Real(j));
        pos.push_back(Real(0));
    }
    std::vector<int> conn;
    auto nid = [&](int i, int j){ return j*(N+1) + i; };
    for (int jq = 0; jq < N; ++jq) for (int iq = 0; iq < N; ++iq) {
        conn.push_back(nid(iq,  jq));
        conn.push_back(nid(iq+1,jq));
        conn.push_back(nid(iq+1,jq+1));
        conn.push_back(nid(iq,  jq+1));
    }
    ContactBucketSearch s;
    s.build(pos.data(), (N+1)*(N+1), conn.data(), N*N, 4);
    CHECK(s.num_segments() == 100, "scaling_100: 100 segments");

    auto pairs = s.find_pairs(Real(0.05));
    // Adjacent quads share edges so many neighbour pairs are expected
    CHECK(!pairs.empty(), "scaling_100: non-empty pair list");
}

// --- Scaling: 1000-element grid (timing sanity check) -----------------------
static void test_scaling_1000()
{
    const int N = 32; // 32x32 = 1024 quads
    std::vector<Real> pos;
    for (int j = 0; j <= N; ++j) for (int i = 0; i <= N; ++i) {
        pos.push_back(Real(i));
        pos.push_back(Real(j));
        pos.push_back(Real(0));
    }
    std::vector<int> conn;
    auto nid = [&](int i, int j){ return j*(N+1) + i; };
    for (int jq = 0; jq < N; ++jq) for (int iq = 0; iq < N; ++iq) {
        conn.push_back(nid(iq,  jq));
        conn.push_back(nid(iq+1,jq));
        conn.push_back(nid(iq+1,jq+1));
        conn.push_back(nid(iq,  jq+1));
    }

    auto t0 = std::chrono::steady_clock::now();
    ContactBucketSearch s;
    s.build(pos.data(), (N+1)*(N+1), conn.data(), N*N, 4);
    auto pairs = s.find_pairs(Real(0.05));
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CHECK(s.num_segments() == N*N, "scaling_1000: correct segment count");
    CHECK(!pairs.empty(), "scaling_1000: non-empty pairs");
    // Sanity: should complete well under 1 second on any modern CPU
    CHECK(ms < 1000.0, "scaling_1000: completes in < 1 s");
}

// --- BoundingBoxHierarchy basic build & query --------------------------------
static void test_bvh_build_query()
{
    std::vector<AABB> aabbs;
    aabbs.push_back(make_aabb(0,0,0, 1,1,1));
    aabbs.push_back(make_aabb(2,2,2, 3,3,3));
    aabbs.push_back(make_aabb(4,4,4, 5,5,5));
    aabbs.push_back(make_aabb(0.5, 0.5, 0.5, 1.5, 1.5, 1.5)); // overlaps box 0

    BoundingBoxHierarchy bvh;
    bvh.build(aabbs.data(), static_cast<int>(aabbs.size()));

    CHECK(bvh.num_nodes() > 0, "bvh: nodes built");

    // Query near box 0 — should find 0 and 3
    std::unordered_set<int> found;
    bvh.query(make_aabb(0,0,0, 2,2,2), [&](int id){ found.insert(id); });
    CHECK(found.count(0) > 0, "bvh query: finds box 0");
    CHECK(found.count(3) > 0, "bvh query: finds box 3");
    CHECK(found.count(2) == 0, "bvh query: does not find far box 2");
}

// --- BoundingBoxHierarchy: single-element ----------------------------------------
static void test_bvh_single()
{
    std::vector<AABB> aabbs = {make_aabb(0,0,0, 1,1,1)};
    BoundingBoxHierarchy bvh;
    bvh.build(aabbs.data(), 1);
    CHECK(bvh.num_nodes() == 1, "bvh_single: 1 node");

    std::vector<int> found;
    bvh.query(make_aabb(0,0,0, 1,1,1), [&](int id){ found.push_back(id); });
    CHECK(found.size() == 1 && found[0] == 0, "bvh_single: finds the single segment");
}

// --- BoundingBoxHierarchy: no overlap ----------------------------------------
static void test_bvh_no_overlap()
{
    std::vector<AABB> aabbs;
    aabbs.push_back(make_aabb(0,0,0, 1,1,1));
    aabbs.push_back(make_aabb(5,5,5, 6,6,6));
    BoundingBoxHierarchy bvh;
    bvh.build(aabbs.data(), 2);

    std::vector<int> found;
    bvh.query(make_aabb(10,10,10, 11,11,11), [&](int id){ found.push_back(id); });
    CHECK(found.empty(), "bvh_no_overlap: empty far query");
}

// --- BoundingBoxHierarchy: larger set ----------------------------------------
static void test_bvh_larger()
{
    const int N = 20;
    std::vector<AABB> aabbs;
    for (int i = 0; i < N; ++i)
        aabbs.push_back(make_aabb(i*2.0, 0, 0, i*2.0+1, 1, 1));

    BoundingBoxHierarchy bvh;
    bvh.build(aabbs.data(), N);

    // Query box covering the first 5 segments (x in [0,9])
    std::unordered_set<int> found;
    bvh.query(make_aabb(0,0,0, 9,1,1), [&](int id){ found.insert(id); });
    CHECK(static_cast<int>(found.size()) == 5, "bvh_larger: finds exactly 5 segments");
}

// --- ContactSortManager: single interface ------------------------------------
static void test_manager_single_interface()
{
    // 4 master + 4 slave segments, all overlapping
    std::vector<AABB> master(4), slave(4);
    for (int i = 0; i < 4; ++i) {
        master[i] = make_aabb(i*1.0, 0, 0, i*1.0+0.9, 1, 1);
        slave[i]  = make_aabb(i*1.0, 0, 0, i*1.0+0.9, 1, 0.5); // overlaps in z
    }

    ContactSortManager mgr;
    mgr.add_interface(master, slave);
    CHECK(mgr.num_interfaces() == 1, "manager: 1 interface");

    auto results = mgr.search_all(Real(0.1));
    CHECK(static_cast<int>(results.size()) == 1, "manager: 1 result set");
    CHECK(!results[0].empty(), "manager: non-empty pairs");
}

// --- ContactSortManager: two interfaces --------------------------------------
static void test_manager_two_interfaces()
{
    std::vector<AABB> m1 = {make_aabb(0,0,0,1,1,1)};
    std::vector<AABB> s1 = {make_aabb(0.5,0.5,0.5,1.5,1.5,1.5)};
    std::vector<AABB> m2 = {make_aabb(10,10,10,11,11,11)};
    std::vector<AABB> s2 = {make_aabb(10.5,10.5,10.5,11.5,11.5,11.5)};

    ContactSortManager mgr;
    mgr.add_interface(m1, s1);
    mgr.add_interface(m2, s2);
    CHECK(mgr.num_interfaces() == 2, "manager_two: 2 interfaces");

    auto results = mgr.search_all(Real(0));
    CHECK(static_cast<int>(results.size()) == 2, "manager_two: 2 result sets");
    CHECK(!results[0].empty(), "manager_two: interface 0 has pairs");
    CHECK(!results[1].empty(), "manager_two: interface 1 has pairs");
}

// --- ContactSortManager: empty master / slave --------------------------------
static void test_manager_empty_interface()
{
    std::vector<AABB> empty_v;
    std::vector<AABB> one = {make_aabb(0,0,0,1,1,1)};

    ContactSortManager mgr;
    mgr.add_interface(empty_v, one);
    auto results = mgr.search_all(Real(0));
    CHECK(results[0].empty(), "manager_empty: empty master => no pairs");

    ContactSortManager mgr2;
    mgr2.add_interface(one, empty_v);
    auto results2 = mgr2.search_all(Real(0));
    CHECK(results2[0].empty(), "manager_empty: empty slave => no pairs");
}

// --- ContactSortManager: update() is a no-op (doesn't crash) ----------------
static void test_manager_update()
{
    std::vector<AABB> m = {make_aabb(0,0,0,1,1,1)};
    std::vector<AABB> s = {make_aabb(0.5,0.5,0.5,1.5,1.5,1.5)};
    ContactSortManager mgr;
    mgr.add_interface(m, s);

    Real dummy_pos[3] = {0, 0, 0};
    mgr.update(dummy_pos); // must not throw or crash
    CHECK(true, "manager_update: update() does not crash");
}

// --- Pair indices are valid --------------------------------------------------
static void test_pair_indices_valid()
{
    const int N = 10;
    std::vector<AABB> aabbs;
    for (int i = 0; i < N; ++i)
        aabbs.push_back(make_aabb(i*0.5, 0, 0, i*0.5+0.6, 1, 1));

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));

    bool all_valid = true;
    for (const auto& p : pairs) {
        if (p.seg1 < 0 || p.seg1 >= N) all_valid = false;
        if (p.seg2 < 0 || p.seg2 >= N) all_valid = false;
        if (p.seg1 >= p.seg2) all_valid = false; // must be unique ordered pairs
    }
    CHECK(all_valid, "pair_indices: all pairs have valid ordered indices");
}

// --- No duplicate pairs ------------------------------------------------------
static void test_no_duplicate_pairs()
{
    std::vector<AABB> aabbs;
    for (int i = 0; i < 8; ++i)
        aabbs.push_back(make_aabb(i*0.5, 0, 0, i*0.5+0.6, 1, 1));

    ContactBucketSearch s;
    s.build(aabbs);
    auto pairs = s.find_pairs(Real(0));

    // Build a set of (s1,s2) and check no duplicates
    std::unordered_set<long long> seen;
    bool no_dups = true;
    for (const auto& p : pairs) {
        long long key = (long long)p.seg1 * 1000 + p.seg2;
        if (seen.count(key)) no_dups = false;
        seen.insert(key);
    }
    CHECK(no_dups, "no_duplicates: all pairs are unique");
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "=== Wave 43: Contact Bucket Sort Tests ===\n\n";

    // AABB tests
    test_aabb_construction();
    test_aabb_expand_point();
    test_aabb_expand_tol();
    test_aabb_overlaps();
    test_aabb_merge();
    test_aabb_centre();
    test_aabb_surface_area();

    // BucketSort3D tests
    test_bucket_insert_query_basic();
    test_bucket_clear();
    test_bucket_clamping();
    test_bucket_multi_range_query();

    // ContactBucketSearch tests
    test_contact_search_quads();
    test_contact_no_false_negatives();
    test_contact_far_apart();
    test_contact_coincident();
    test_contact_touching();
    test_contact_gap_sign();

    // Scaling tests
    test_scaling_100();
    test_scaling_1000();

    // BoundingBoxHierarchy tests
    test_bvh_build_query();
    test_bvh_single();
    test_bvh_no_overlap();
    test_bvh_larger();

    // ContactSortManager tests
    test_manager_single_interface();
    test_manager_two_interfaces();
    test_manager_empty_interface();
    test_manager_update();

    // Integrity tests
    test_pair_indices_valid();
    test_no_duplicate_pairs();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
