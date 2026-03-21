#pragma once

/**
 * @file contact_wave43.hpp
 * @brief Wave 43: Spatial Bucket Sort for O(N) Contact Search
 *
 * Provides a uniform spatial hash grid (BucketSort3D) that reduces contact
 * pair detection from O(N²) to O(N) for uniformly distributed geometry.
 *
 * Sub-modules:
 * - AABB                   — Axis-aligned bounding box with GPU-compatible helpers
 * - BucketSort3D           — Uniform spatial hash grid (host-side, std::vector)
 * - ContactBucketSearch    — Surface-segment AABB builder + pair finder
 * - BoundingBoxHierarchy   — Optional AABB tree for secondary broad-phase
 * - ContactSortManager     — Multi-interface search orchestrator
 *
 * References:
 * - Mirtich (1996) "Impulse-based dynamic simulation of rigid body systems"
 * - Bergen (2004) "Collision Detection in Interactive 3D Environments"
 * - Ericson (2005) "Real-Time Collision Detection"
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

// Kokkos macro shim — falls back to nothing when Kokkos is not present
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Wave 43 internal helpers
// ============================================================================

namespace wave43_detail {

KOKKOS_INLINE_FUNCTION
Real w43_min(Real a, Real b) { return a < b ? a : b; }

KOKKOS_INLINE_FUNCTION
Real w43_max(Real a, Real b) { return a > b ? a : b; }

} // namespace wave43_detail

// ============================================================================
// AABB — Axis-Aligned Bounding Box
// ============================================================================

/**
 * @brief Axis-aligned bounding box.
 *
 * All methods are KOKKOS_INLINE_FUNCTION so they can be called from both
 * host and device code.
 */
struct AABB {
    Real min_pt[3];
    Real max_pt[3];

    /// Default constructor — degenerate box at origin
    KOKKOS_INLINE_FUNCTION
    AABB() {
        for (int i = 0; i < 3; ++i) {
            min_pt[i] = Real(0);
            max_pt[i] = Real(0);
        }
    }

    /// Construct from explicit min/max corners
    KOKKOS_INLINE_FUNCTION
    AABB(const Real lo[3], const Real hi[3]) {
        for (int i = 0; i < 3; ++i) {
            min_pt[i] = lo[i];
            max_pt[i] = hi[i];
        }
    }

    /// Construct an "inside-out" box suitable for incremental expansion
    KOKKOS_INLINE_FUNCTION
    static AABB empty() {
        AABB b;
        for (int i = 0; i < 3; ++i) {
            b.min_pt[i] =  std::numeric_limits<Real>::max();
            b.max_pt[i] = -std::numeric_limits<Real>::max();
        }
        return b;
    }

    /// Expand this box to contain point p
    KOKKOS_INLINE_FUNCTION
    void expand(const Real p[3]) {
        for (int i = 0; i < 3; ++i) {
            min_pt[i] = wave43_detail::w43_min(min_pt[i], p[i]);
            max_pt[i] = wave43_detail::w43_max(max_pt[i], p[i]);
        }
    }

    /// Expand each face outward by tol (uniform padding)
    KOKKOS_INLINE_FUNCTION
    void expand(Real tol) {
        for (int i = 0; i < 3; ++i) {
            min_pt[i] -= tol;
            max_pt[i] += tol;
        }
    }

    /// Return true if this box overlaps other (touching counts)
    KOKKOS_INLINE_FUNCTION
    bool overlaps(const AABB& other) const {
        for (int i = 0; i < 3; ++i) {
            if (max_pt[i] < other.min_pt[i]) return false;
            if (min_pt[i] > other.max_pt[i]) return false;
        }
        return true;
    }

    /// Merge: return the smallest box containing both this and other
    KOKKOS_INLINE_FUNCTION
    AABB merge(const AABB& other) const {
        AABB result;
        for (int i = 0; i < 3; ++i) {
            result.min_pt[i] = wave43_detail::w43_min(min_pt[i], other.min_pt[i]);
            result.max_pt[i] = wave43_detail::w43_max(max_pt[i], other.max_pt[i]);
        }
        return result;
    }

    /// Centre of the box
    KOKKOS_INLINE_FUNCTION
    void centre(Real c[3]) const {
        for (int i = 0; i < 3; ++i)
            c[i] = Real(0.5) * (min_pt[i] + max_pt[i]);
    }

    /// Half-extents
    KOKKOS_INLINE_FUNCTION
    void half_extents(Real h[3]) const {
        for (int i = 0; i < 3; ++i)
            h[i] = Real(0.5) * (max_pt[i] - min_pt[i]);
    }

    /// Surface area (for SAH heuristics)
    KOKKOS_INLINE_FUNCTION
    Real surface_area() const {
        Real dx = max_pt[0] - min_pt[0];
        Real dy = max_pt[1] - min_pt[1];
        Real dz = max_pt[2] - min_pt[2];
        return Real(2) * (dx*dy + dy*dz + dz*dx);
    }

    /// Returns true when the box has non-negative extents in every axis
    KOKKOS_INLINE_FUNCTION
    bool is_valid() const {
        for (int i = 0; i < 3; ++i)
            if (min_pt[i] > max_pt[i]) return false;
        return true;
    }
};

// ============================================================================
// BucketSort3D — Uniform spatial hash grid
// ============================================================================

/**
 * @brief Uniform spatial hash grid for O(N) broad-phase contact search.
 *
 * Divides the domain AABB into a regular grid of voxel buckets. Each
 * inserted object is hashed to the bucket containing its centre; queries
 * iterate over all buckets that overlap the query AABB and invoke a
 * user-supplied callback for every stored id.
 *
 * Host-side only — uses std::vector<std::vector<int>> for bucket storage.
 */
class BucketSort3D {
public:
    /**
     * @param domain    Bounding box of the entire problem domain
     * @param bucket_sz Desired edge length of each voxel bucket
     */
    BucketSort3D(const AABB& domain, Real bucket_sz)
        : domain_(domain), bucket_size_(bucket_sz)
    {
        if (bucket_sz <= Real(0))
            throw std::invalid_argument("BucketSort3D: bucket_size must be > 0");

        for (int i = 0; i < 3; ++i) {
            Real span = domain_.max_pt[i] - domain_.min_pt[i];
            dims_[i] = std::max(1, static_cast<int>(std::ceil(span / bucket_sz)));
        }
        buckets_.resize(static_cast<std::size_t>(dims_[0]) *
                        static_cast<std::size_t>(dims_[1]) *
                        static_cast<std::size_t>(dims_[2]));
    }

    /// Remove all stored ids from every bucket
    void clear() {
        for (auto& b : buckets_) b.clear();
    }

    /**
     * @brief Insert id into the bucket that contains the centre of box.
     *
     * If the centre falls outside the domain the id is clamped to the
     * nearest bucket rather than discarded.
     */
    void insert(int id, const AABB& box) {
        Real c[3];
        box.centre(c);
        int ix = clamp_idx(static_cast<int>(std::floor(
                    (c[0] - domain_.min_pt[0]) / bucket_size_)), dims_[0]);
        int iy = clamp_idx(static_cast<int>(std::floor(
                    (c[1] - domain_.min_pt[1]) / bucket_size_)), dims_[1]);
        int iz = clamp_idx(static_cast<int>(std::floor(
                    (c[2] - domain_.min_pt[2]) / bucket_size_)), dims_[2]);
        buckets_[hash(ix, iy, iz)].push_back(id);
    }

    /**
     * @brief Query all ids whose bucket overlaps query_box.
     *
     * @param query_box  The AABB to test against
     * @param callback   void callback(int id) called for each candidate
     */
    template<typename Callback>
    void query(const AABB& query_box, Callback&& callback) const {
        // Determine bucket index range overlapping query_box
        int x0 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.min_pt[0] - domain_.min_pt[0]) / bucket_size_)), dims_[0]);
        int x1 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.max_pt[0] - domain_.min_pt[0]) / bucket_size_)), dims_[0]);
        int y0 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.min_pt[1] - domain_.min_pt[1]) / bucket_size_)), dims_[1]);
        int y1 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.max_pt[1] - domain_.min_pt[1]) / bucket_size_)), dims_[1]);
        int z0 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.min_pt[2] - domain_.min_pt[2]) / bucket_size_)), dims_[2]);
        int z1 = clamp_idx(static_cast<int>(std::floor(
                    (query_box.max_pt[2] - domain_.min_pt[2]) / bucket_size_)), dims_[2]);

        for (int iz = z0; iz <= z1; ++iz)
            for (int iy = y0; iy <= y1; ++iy)
                for (int ix = x0; ix <= x1; ++ix)
                    for (int cand : buckets_[hash(ix, iy, iz)])
                        callback(cand);
    }

    /// Total number of buckets
    int num_buckets() const {
        return dims_[0] * dims_[1] * dims_[2];
    }

    /// Grid dimensions per axis
    void grid_dims(int out[3]) const {
        out[0] = dims_[0]; out[1] = dims_[1]; out[2] = dims_[2];
    }

    /// Total number of stored ids across all buckets
    int total_entries() const {
        int n = 0;
        for (const auto& b : buckets_) n += static_cast<int>(b.size());
        return n;
    }

private:
    AABB   domain_;
    Real   bucket_size_;
    int    dims_[3];
    std::vector<std::vector<int>> buckets_;

    /// Flat bucket index from 3D grid coordinates
    inline int hash(int ix, int iy, int iz) const {
        return iz * dims_[0] * dims_[1] + iy * dims_[0] + ix;
    }

    /// Clamp grid index to [0, dim-1]
    static inline int clamp_idx(int i, int dim) {
        if (i < 0) return 0;
        if (i >= dim) return dim - 1;
        return i;
    }
};

// ============================================================================
// ContactPair
// ============================================================================

/// A candidate contact pair returned by broad-phase search
struct ContactPair {
    int  seg1;         ///< Index of first surface segment
    int  seg2;         ///< Index of second surface segment
    Real gap_estimate; ///< Approximate gap (negative = penetration)
};

// ============================================================================
// ContactBucketSearch — builds segment AABBs and finds pairs
// ============================================================================

/**
 * @brief Builds AABBs for all surface segments and detects candidate contact
 *        pairs using a BucketSort3D broad-phase.
 *
 * A "segment" is a surface face (tri or quad) defined by its node indices.
 * Supports Tri3 (3 nodes) and Quad4 (4 nodes) surface facets; the facet
 * type is inferred from the element_type argument or the per-segment
 * node count.
 */
class ContactBucketSearch {
public:
    ContactBucketSearch() = default;

    /**
     * @brief Build AABB list for surface segments from nodal positions.
     *
     * @param positions     Flat array of node positions: [x0,y0,z0, x1,y1,z1, ...]
     * @param num_nodes     Number of nodes
     * @param connectivity  Flat segment connectivity.  Each segment consumes
     *                      nodes_per_seg consecutive entries.
     * @param num_segments  Number of surface segments
     * @param nodes_per_seg Nodes per segment: 3 (tri) or 4 (quad)
     * @param gap_padding   Optional extra padding added to every AABB
     */
    void build(const Real* positions, int num_nodes,
               const int*  connectivity, int num_segments,
               int nodes_per_seg = 4,
               Real gap_padding  = Real(0))
    {
        (void)num_nodes; // not needed for AABB construction but kept for API clarity
        num_segments_  = num_segments;
        nodes_per_seg_ = nodes_per_seg;
        segment_aabbs_.resize(static_cast<std::size_t>(num_segments));

        // Compute per-segment AABB
        AABB domain = AABB::empty();
        for (int s = 0; s < num_segments; ++s) {
            AABB box = AABB::empty();
            for (int n = 0; n < nodes_per_seg; ++n) {
                int nid = connectivity[s * nodes_per_seg + n];
                const Real* p = positions + 3 * nid;
                box.expand(p);
            }
            if (gap_padding > Real(0)) box.expand(gap_padding);
            segment_aabbs_[static_cast<std::size_t>(s)] = box;
            domain = domain.merge(box);
        }
        domain_ = domain;
    }

    /**
     * @brief Build from pre-computed per-segment AABBs (convenience overload).
     */
    void build(const std::vector<AABB>& aabbs) {
        num_segments_  = static_cast<int>(aabbs.size());
        nodes_per_seg_ = 0;
        segment_aabbs_ = aabbs;
        domain_ = AABB::empty();
        for (const auto& b : aabbs) domain_ = domain_.merge(b);
    }

    /**
     * @brief Find all candidate contact pairs whose AABBs overlap.
     *
     * @param gap_tolerance  Extra tolerance added to query box (widens search)
     * @return               Vector of candidate ContactPairs (broad-phase only)
     */
    std::vector<ContactPair> find_pairs(Real gap_tolerance = Real(0)) const {
        if (num_segments_ == 0) return {};

        // Choose bucket size: default to ~2x the average segment AABB diagonal
        Real avg_diag = compute_avg_diagonal();
        Real bucket_sz = Real(2) * avg_diag;
        if (bucket_sz <= Real(0)) bucket_sz = Real(1);

        // Expand domain slightly to avoid edge-bucket ambiguity
        AABB expanded_domain = domain_;
        expanded_domain.expand(bucket_sz * Real(0.01));

        BucketSort3D grid(expanded_domain, bucket_sz);

        // Insert all segments
        for (int s = 0; s < num_segments_; ++s)
            grid.insert(s, segment_aabbs_[static_cast<std::size_t>(s)]);

        // Query: for each segment, find candidates
        std::vector<ContactPair> pairs;
        for (int s = 0; s < num_segments_; ++s) {
            AABB query = segment_aabbs_[static_cast<std::size_t>(s)];
            if (gap_tolerance > Real(0)) query.expand(gap_tolerance);

            grid.query(query, [&](int cand) {
                // Only unique pairs (s < cand) and skip self
                if (cand <= s) return;
                const AABB& other = segment_aabbs_[static_cast<std::size_t>(cand)];
                if (!query.overlaps(other)) return;

                // Estimate gap as centre-to-centre distance minus half-extents sum
                Real gap = estimate_gap(segment_aabbs_[static_cast<std::size_t>(s)], other);
                pairs.push_back({s, cand, gap});
            });
        }
        return pairs;
    }

    const AABB& domain() const { return domain_; }
    int  num_segments() const { return num_segments_; }
    const std::vector<AABB>& segment_aabbs() const { return segment_aabbs_; }

private:
    int                num_segments_  = 0;
    int                nodes_per_seg_ = 4;
    std::vector<AABB>  segment_aabbs_;
    AABB               domain_;

    Real compute_avg_diagonal() const {
        if (segment_aabbs_.empty()) return Real(1);
        Real sum = Real(0);
        for (const auto& b : segment_aabbs_) {
            Real dx = b.max_pt[0] - b.min_pt[0];
            Real dy = b.max_pt[1] - b.min_pt[1];
            Real dz = b.max_pt[2] - b.min_pt[2];
            sum += std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        return sum / static_cast<Real>(segment_aabbs_.size());
    }

    static Real estimate_gap(const AABB& a, const AABB& b) {
        // Gap = max separation across all axes (negative means overlap)
        Real gap = -std::numeric_limits<Real>::max();
        for (int i = 0; i < 3; ++i) {
            Real sep = wave43_detail::w43_max(a.min_pt[i] - b.max_pt[i],
                                              b.min_pt[i] - a.max_pt[i]);
            gap = wave43_detail::w43_max(gap, sep);
        }
        return gap;
    }
};

// ============================================================================
// BoundingBoxHierarchy — simple AABB tree (optional secondary broad-phase)
// ============================================================================

/**
 * @brief Binary AABB tree for hierarchical broad-phase contact queries.
 *
 * Construction uses a top-down median-split strategy along the longest axis.
 * Leaf nodes correspond to individual surface segments.
 */
class BoundingBoxHierarchy {
public:
    struct Node {
        AABB  box;
        int   left  = -1;  ///< Index of left child (-1 = leaf)
        int   right = -1;  ///< Index of right child (-1 = leaf)
        int   seg_id = -1; ///< Segment id (leaf nodes only)
    };

    BoundingBoxHierarchy() = default;

    /// Build the BVH from a flat array of per-segment AABBs
    void build(const AABB* aabbs, int num) {
        nodes_.clear();
        if (num <= 0) return;
        std::vector<int> indices(static_cast<std::size_t>(num));
        for (int i = 0; i < num; ++i) indices[static_cast<std::size_t>(i)] = i;
        aabbs_ = aabbs;
        build_recursive(indices, 0, num);
    }

    /// Query: invoke callback(seg_id) for every leaf whose AABB overlaps query_box
    template<typename Callback>
    void query(const AABB& query_box, Callback&& callback) const {
        if (nodes_.empty()) return;
        query_recursive(0, query_box, callback);
    }

    int num_nodes() const { return static_cast<int>(nodes_.size()); }

private:
    std::vector<Node>  nodes_;
    const AABB*        aabbs_ = nullptr;

    int build_recursive(std::vector<int>& indices, int start, int end) {
        int node_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();

        // Compute bounding box for this range
        AABB box = AABB::empty();
        for (int i = start; i < end; ++i)
            box = box.merge(aabbs_[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])]);
        nodes_[static_cast<std::size_t>(node_idx)].box = box;

        if (end - start == 1) {
            // Leaf
            nodes_[static_cast<std::size_t>(node_idx)].seg_id = indices[static_cast<std::size_t>(start)];
            return node_idx;
        }

        // Split along longest axis at median centroid
        int axis = longest_axis(box);
        int mid  = (start + end) / 2;
        std::nth_element(
            indices.begin() + start,
            indices.begin() + mid,
            indices.begin() + end,
            [&](int a, int b) {
                Real ca[3], cb[3];
                aabbs_[static_cast<std::size_t>(a)].centre(ca);
                aabbs_[static_cast<std::size_t>(b)].centre(cb);
                return ca[axis] < cb[axis];
            });

        int left_child  = build_recursive(indices, start, mid);
        int right_child = build_recursive(indices, mid,   end);
        // node_idx position may have been invalidated by emplace_back — re-fetch
        nodes_[static_cast<std::size_t>(node_idx)].left  = left_child;
        nodes_[static_cast<std::size_t>(node_idx)].right = right_child;
        return node_idx;
    }

    template<typename Callback>
    void query_recursive(int node_idx, const AABB& q, Callback&& cb) const {
        const Node& node = nodes_[static_cast<std::size_t>(node_idx)];
        if (!node.box.overlaps(q)) return;
        if (node.left == -1) {
            // Leaf
            cb(node.seg_id);
            return;
        }
        query_recursive(node.left,  q, cb);
        query_recursive(node.right, q, cb);
    }

    static int longest_axis(const AABB& b) {
        Real dx = b.max_pt[0] - b.min_pt[0];
        Real dy = b.max_pt[1] - b.min_pt[1];
        Real dz = b.max_pt[2] - b.min_pt[2];
        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }
};

// ============================================================================
// ContactSortManager — multi-interface search orchestrator
// ============================================================================

/**
 * @brief Manages spatial acceleration structures for multiple contact
 *        interfaces and returns per-interface candidate pair lists.
 */
class ContactSortManager {
public:
    struct Interface {
        std::vector<AABB> master_aabbs;
        std::vector<AABB> slave_aabbs;
    };

    ContactSortManager() = default;

    /**
     * @brief Register a contact interface with pre-computed segment AABBs.
     *
     * @param master_segments  AABBs for master surface segments
     * @param slave_segments   AABBs for slave surface segments
     */
    void add_interface(const std::vector<AABB>& master_segments,
                       const std::vector<AABB>& slave_segments)
    {
        interfaces_.push_back({master_segments, slave_segments});
    }

    /**
     * @brief Rebuild spatial acceleration structures after geometry update.
     *
     * @param positions  Flat nodal position array [x0,y0,z0, ...]
     *                   (Currently a hook for future dynamic updates;
     *                    when AABBs are pre-computed this is a no-op.)
     */
    void update(const Real* /*positions*/) {
        // Spatial structures are rebuilt lazily in search_all.
        // This hook exists for callers that re-compute AABBs externally.
    }

    /**
     * @brief Search all interfaces for candidate contact pairs.
     *
     * @param gap_tolerance  AABB query padding (same for all interfaces)
     * @return               One vector of ContactPairs per registered interface
     */
    std::vector<std::vector<ContactPair>>
    search_all(Real gap_tolerance = Real(0)) const
    {
        std::vector<std::vector<ContactPair>> results;
        results.reserve(interfaces_.size());

        for (const auto& iface : interfaces_) {
            results.push_back(search_interface(iface, gap_tolerance));
        }
        return results;
    }

    int num_interfaces() const { return static_cast<int>(interfaces_.size()); }

private:
    std::vector<Interface> interfaces_;

    static std::vector<ContactPair>
    search_interface(const Interface& iface, Real gap_tol)
    {
        const int nm = static_cast<int>(iface.master_aabbs.size());
        const int ns = static_cast<int>(iface.slave_aabbs.size());
        if (nm == 0 || ns == 0) return {};

        // Build domain over both surfaces
        AABB domain = AABB::empty();
        for (const auto& b : iface.master_aabbs) domain = domain.merge(b);
        for (const auto& b : iface.slave_aabbs)  domain = domain.merge(b);

        // Choose bucket size
        Real avg_diag = Real(0);
        for (const auto& b : iface.master_aabbs) {
            Real dx = b.max_pt[0]-b.min_pt[0];
            Real dy = b.max_pt[1]-b.min_pt[1];
            Real dz = b.max_pt[2]-b.min_pt[2];
            avg_diag += std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        avg_diag /= static_cast<Real>(nm);
        Real bucket_sz = Real(2) * avg_diag;
        if (bucket_sz <= Real(0)) bucket_sz = Real(1);

        AABB expanded = domain;
        expanded.expand(bucket_sz * Real(0.01));

        // Insert slave segments into grid
        BucketSort3D grid(expanded, bucket_sz);
        for (int s = 0; s < ns; ++s)
            grid.insert(s, iface.slave_aabbs[static_cast<std::size_t>(s)]);

        // Query with each master segment
        std::vector<ContactPair> pairs;
        for (int m = 0; m < nm; ++m) {
            AABB q = iface.master_aabbs[static_cast<std::size_t>(m)];
            if (gap_tol > Real(0)) q.expand(gap_tol);

            grid.query(q, [&](int s) {
                const AABB& sb = iface.slave_aabbs[static_cast<std::size_t>(s)];
                if (!q.overlaps(sb)) return;

                Real gap = -std::numeric_limits<Real>::max();
                for (int i = 0; i < 3; ++i) {
                    Real sep = wave43_detail::w43_max(
                        iface.master_aabbs[static_cast<std::size_t>(m)].min_pt[i] - sb.max_pt[i],
                        sb.min_pt[i] - iface.master_aabbs[static_cast<std::size_t>(m)].max_pt[i]);
                    gap = wave43_detail::w43_max(gap, sep);
                }
                pairs.push_back({m, s, gap});
            });
        }
        return pairs;
    }
};

} // namespace fem
} // namespace nxs
