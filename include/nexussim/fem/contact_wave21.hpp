#pragma once

/**
 * @file contact_wave21.hpp
 * @brief Wave 21: Contact Refinement Features for NexusSim
 *
 * Sub-modules:
 * - 21g: AutomaticContactDetection — Surface extraction from volume mesh + bucket search
 * - 21h: Contact2D — Plane strain and axisymmetric node-to-segment contact
 * - 21i: SPHContact — Particle-to-surface contact for SPH-FEM coupling
 * - 21j: AirbagFabricContact — Fabric self-contact with porosity/leak-through
 * - 21k: ContactHeatGeneration — Friction-induced heat at contact interface
 * - 21l: MortarContactFriction — Mortar contact with Coulomb friction
 *
 * References:
 * - Wriggers (2006) "Computational Contact Mechanics"
 * - Hallquist (2006) "LS-DYNA Theory Manual"
 * - Puso & Laursen (2004) "A mortar segment-to-segment contact method"
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Contact utility helpers
// ============================================================================

namespace contact_detail {

/// Dot product of two 3-vectors
KOKKOS_INLINE_FUNCTION
Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Cross product c = a x b
KOKKOS_INLINE_FUNCTION
void cross3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

/// Euclidean norm
KOKKOS_INLINE_FUNCTION
Real norm3(const Real v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/// Normalize in-place; returns original length
KOKKOS_INLINE_FUNCTION
Real normalize3(Real v[3]) {
    Real len = norm3(v);
    if (len > 1.0e-30) {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
    return len;
}

/// Subtraction c = a - b
KOKKOS_INLINE_FUNCTION
void sub3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0]-b[0]; c[1] = a[1]-b[1]; c[2] = a[2]-b[2];
}

/// Point-to-triangle closest point and squared distance.
/// Uses the parametric projection method (Eberly).
KOKKOS_INLINE_FUNCTION
Real point_triangle_dist2(const Real p[3],
                                  const Real v0[3], const Real v1[3], const Real v2[3],
                                  Real closest[3]) {
    Real e0[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    Real e1[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    Real d[3]  = { v0[0]-p[0], v0[1]-p[1], v0[2]-p[2] };

    Real a = dot3(e0, e0);
    Real b = dot3(e0, e1);
    Real c_val = dot3(e1, e1);
    Real dd = dot3(e0, d);
    Real e = dot3(e1, d);

    Real det = a * c_val - b * b;
    Real s = b * e - c_val * dd;
    Real t = b * dd - a * e;

    if (det < 1.0e-30) {
        s = 0.0; t = 0.0;
    } else {
        if (s + t <= det) {
            if (s < 0.0) {
                if (t < 0.0) {
                    s = (-dd >= 0.0) ? 0.0 : ((-dd >= a) ? 1.0 : -dd / a);
                    t = 0.0;
                } else {
                    s = 0.0;
                    t = (e >= 0.0) ? 0.0 : ((-e >= c_val) ? 1.0 : -e / c_val);
                }
            } else if (t < 0.0) {
                s = (-dd >= 0.0) ? 0.0 : ((-dd >= a) ? 1.0 : -dd / a);
                t = 0.0;
            } else {
                Real inv_det = 1.0 / det;
                s *= inv_det;
                t *= inv_det;
            }
        } else {
            if (s < 0.0) {
                s = 0.0;
                t = (e >= 0.0) ? 0.0 : ((-e >= c_val) ? 1.0 : -e / c_val);
            } else if (t < 0.0) {
                s = (-dd >= 0.0) ? 0.0 : ((-dd >= a) ? 1.0 : -dd / a);
                t = 0.0;
            } else {
                Real numer = (c_val + e) - (b + dd);
                if (numer <= 0.0) {
                    s = 0.0;
                } else {
                    Real denom = a - 2.0*b + c_val;
                    s = (numer >= denom) ? 1.0 : numer / denom;
                }
                t = 1.0 - s;
            }
        }
    }

    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;
    if (t < 0.0) t = 0.0; if (t > 1.0) t = 1.0;
    if (s + t > 1.0) { Real sc = 1.0 / (s + t); s *= sc; t *= sc; }

    closest[0] = v0[0] + s * e0[0] + t * e1[0];
    closest[1] = v0[1] + s * e0[1] + t * e1[1];
    closest[2] = v0[2] + s * e0[2] + t * e1[2];

    Real diff[3] = { p[0]-closest[0], p[1]-closest[1], p[2]-closest[2] };
    return dot3(diff, diff);
}

} // namespace contact_detail

// ============================================================================
// 21g: AutomaticContactDetection — Surface Extraction + Bucket Search
// ============================================================================

/**
 * @brief Automatic contact detection: extract surface from volume mesh,
 *        build contact pairs via spatial bucket search.
 *
 * Algorithm for surface extraction:
 * 1. Build face-to-element map from volume connectivity
 * 2. Exterior faces are those with exactly one parent element
 * 3. Orient outward normals using cross product of edge vectors
 *
 * Spatial bucket search:
 * 1. Partition space into uniform grid of buckets
 * 2. Insert face centroids into buckets
 * 3. For each slave node, query nearby buckets for candidate faces
 */
class AutomaticContactDetection {
public:
    /// Surface face: 3 or 4 node IDs (tri or quad), with parent element
    struct SurfaceFace {
        int nodes[4];       ///< Node IDs (nodes[3] = -1 for tri)
        int parent_element; ///< Parent volume element
        int num_nodes;      ///< 3 for tri, 4 for quad
        Real normal[3];     ///< Outward unit normal
        Real centroid[3];   ///< Face centroid

        SurfaceFace() : parent_element(-1), num_nodes(0) {
            for (int i = 0; i < 4; ++i) nodes[i] = -1;
            normal[0] = normal[1] = normal[2] = 0.0;
            centroid[0] = centroid[1] = centroid[2] = 0.0;
        }
    };

    /// Contact pair: slave node + master face
    struct ContactPair {
        int slave_node;
        int master_face;
        Real gap;           ///< Signed gap (negative = penetration)
        Real normal[3];     ///< Contact normal

        ContactPair() : slave_node(-1), master_face(-1), gap(0.0) {
            normal[0] = normal[1] = normal[2] = 0.0;
        }
    };

    AutomaticContactDetection() = default;

    /**
     * @brief Extract exterior surface faces from a volume mesh.
     *
     * For hex elements (8-node connectivity):
     *   Face 0: {0,1,2,3}, Face 1: {4,5,6,7}, Face 2: {0,1,5,4},
     *   Face 3: {2,3,7,6}, Face 4: {0,3,7,4}, Face 5: {1,2,6,5}
     *
     * For tet elements (4-node connectivity):
     *   Face 0: {0,1,2}, Face 1: {0,1,3}, Face 2: {1,2,3}, Face 3: {0,2,3}
     *
     * A face is exterior if it appears in exactly one element.
     *
     * @param connectivity Element connectivity [num_elements][max_nodes_per_elem]
     * @param nodes_per_elem Nodes per element (4=tet, 8=hex)
     * @param num_elements Number of volume elements
     * @param node_coords Node coordinates [num_nodes][3]
     * @param surface_faces Output: extracted exterior faces
     * @return Number of surface faces found
     */
    int extract_surface(const int connectivity[][8],
                        int nodes_per_elem,
                        int num_elements,
                        const Real node_coords[][3],
                        std::vector<SurfaceFace>& surface_faces) const {
        struct FaceKey {
            int n[4];
            bool operator==(const FaceKey& o) const {
                return n[0]==o.n[0] && n[1]==o.n[1] && n[2]==o.n[2] && n[3]==o.n[3];
            }
        };

        struct FaceKeyHash {
            size_t operator()(const FaceKey& k) const {
                size_t h = 0;
                for (int i = 0; i < 4; ++i)
                    h ^= std::hash<int>{}(k.n[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };

        struct FaceInfo {
            int elem;
            int face_nodes[4];
            int num_face_nodes;
            int count;
        };

        std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> face_map;

        static const int hex_faces[6][4] = {
            {0,1,2,3}, {4,5,6,7}, {0,1,5,4}, {2,3,7,6}, {0,3,7,4}, {1,2,6,5}
        };
        static const int tet_faces[4][3] = {
            {0,1,2}, {0,1,3}, {1,2,3}, {0,2,3}
        };

        for (int e = 0; e < num_elements; ++e) {
            int nfaces = (nodes_per_elem == 8) ? 6 : 4;
            for (int f = 0; f < nfaces; ++f) {
                FaceKey key;
                int face_nodes[4] = {-1, -1, -1, -1};
                int nfn;

                if (nodes_per_elem == 8) {
                    nfn = 4;
                    for (int i = 0; i < 4; ++i) {
                        face_nodes[i] = connectivity[e][hex_faces[f][i]];
                        key.n[i] = face_nodes[i];
                    }
                } else {
                    nfn = 3;
                    for (int i = 0; i < 3; ++i) {
                        face_nodes[i] = connectivity[e][tet_faces[f][i]];
                        key.n[i] = face_nodes[i];
                    }
                    key.n[3] = -1;
                }

                std::sort(key.n, key.n + nfn);

                auto it = face_map.find(key);
                if (it == face_map.end()) {
                    FaceInfo fi;
                    fi.elem = e;
                    for (int i = 0; i < 4; ++i) fi.face_nodes[i] = face_nodes[i];
                    fi.num_face_nodes = nfn;
                    fi.count = 1;
                    face_map[key] = fi;
                } else {
                    it->second.count++;
                }
            }
        }

        surface_faces.clear();
        for (auto& [key, fi] : face_map) {
            if (fi.count != 1) continue;

            SurfaceFace sf;
            sf.parent_element = fi.elem;
            sf.num_nodes = fi.num_face_nodes;
            for (int i = 0; i < fi.num_face_nodes; ++i) sf.nodes[i] = fi.face_nodes[i];

            for (int i = 0; i < sf.num_nodes; ++i) {
                int n = sf.nodes[i];
                sf.centroid[0] += node_coords[n][0];
                sf.centroid[1] += node_coords[n][1];
                sf.centroid[2] += node_coords[n][2];
            }
            Real inv_n = 1.0 / static_cast<Real>(sf.num_nodes);
            sf.centroid[0] *= inv_n;
            sf.centroid[1] *= inv_n;
            sf.centroid[2] *= inv_n;

            if (sf.num_nodes >= 3) {
                Real e0[3], e1[3];
                int n0 = sf.nodes[0], n1 = sf.nodes[1], n2 = sf.nodes[2];
                contact_detail::sub3(node_coords[n1], node_coords[n0], e0);
                contact_detail::sub3(node_coords[n2], node_coords[n0], e1);
                contact_detail::cross3(e0, e1, sf.normal);
                contact_detail::normalize3(sf.normal);
            }

            surface_faces.push_back(sf);
        }

        return static_cast<int>(surface_faces.size());
    }

    /**
     * @brief Build contact pairs between two surfaces using bucket search.
     *
     * @param surface1 Slave surface faces
     * @param surface2 Master surface faces
     * @param node_coords Node coordinates
     * @param search_radius Maximum search distance
     * @param pairs Output: detected contact pairs
     * @return Number of contact pairs
     */
    int build_contact_pairs(const std::vector<SurfaceFace>& surface1,
                            const std::vector<SurfaceFace>& surface2,
                            const Real node_coords[][3],
                            Real search_radius,
                            std::vector<ContactPair>& pairs) const {
        pairs.clear();
        if (surface2.empty()) return 0;

        // Build spatial bucket grid for master surface
        Real min_pt[3] = {1.0e30, 1.0e30, 1.0e30};
        Real max_pt[3] = {-1.0e30, -1.0e30, -1.0e30};

        for (auto& sf : surface2) {
            for (int d = 0; d < 3; ++d) {
                min_pt[d] = std::min(min_pt[d], sf.centroid[d] - search_radius);
                max_pt[d] = std::max(max_pt[d], sf.centroid[d] + search_radius);
            }
        }

        Real bucket_size = 2.0 * search_radius;
        if (bucket_size < 1.0e-30) bucket_size = 1.0;

        int nx = std::max(1, std::min(256, static_cast<int>((max_pt[0]-min_pt[0])/bucket_size)+1));
        int ny = std::max(1, std::min(256, static_cast<int>((max_pt[1]-min_pt[1])/bucket_size)+1));
        int nz = std::max(1, std::min(256, static_cast<int>((max_pt[2]-min_pt[2])/bucket_size)+1));

        std::unordered_map<int64_t, std::vector<int>> buckets;

        auto bucket_key = [&](Real x, Real y, Real z) -> int64_t {
            int ix = std::max(0, std::min(nx-1, static_cast<int>((x-min_pt[0])/bucket_size)));
            int iy = std::max(0, std::min(ny-1, static_cast<int>((y-min_pt[1])/bucket_size)));
            int iz = std::max(0, std::min(nz-1, static_cast<int>((z-min_pt[2])/bucket_size)));
            return static_cast<int64_t>(ix)*ny*nz + static_cast<int64_t>(iy)*nz + iz;
        };

        for (int f = 0; f < static_cast<int>(surface2.size()); ++f) {
            int64_t key = bucket_key(surface2[f].centroid[0],
                                     surface2[f].centroid[1],
                                     surface2[f].centroid[2]);
            buckets[key].push_back(f);
        }

        // Collect unique slave nodes
        std::unordered_set<int> slave_nodes;
        for (auto& sf : surface1) {
            for (int i = 0; i < sf.num_nodes; ++i)
                slave_nodes.insert(sf.nodes[i]);
        }

        for (int sn : slave_nodes) {
            Real px = node_coords[sn][0];
            Real py = node_coords[sn][1];
            Real pz = node_coords[sn][2];

            int bx = static_cast<int>((px - min_pt[0]) / bucket_size);
            int by = static_cast<int>((py - min_pt[1]) / bucket_size);
            int bz = static_cast<int>((pz - min_pt[2]) / bucket_size);

            Real best_dist2 = search_radius * search_radius;
            int best_face = -1;
            Real best_normal[3] = {0, 0, 0};
            Real best_closest[3] = {0, 0, 0};

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int dk = -1; dk <= 1; ++dk) {
                        int ix = bx+di, iy = by+dj, iz = bz+dk;
                        if (ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz) continue;
                        int64_t key = static_cast<int64_t>(ix)*ny*nz
                                    + static_cast<int64_t>(iy)*nz + iz;
                        auto it = buckets.find(key);
                        if (it == buckets.end()) continue;

                        for (int fi : it->second) {
                            const SurfaceFace& mf = surface2[fi];
                            Real closest[3];
                            Real d2 = contact_detail::point_triangle_dist2(
                                node_coords[sn],
                                node_coords[mf.nodes[0]],
                                node_coords[mf.nodes[1]],
                                node_coords[mf.nodes[2]],
                                closest);

                            if (mf.num_nodes == 4) {
                                Real closest2[3];
                                Real d2_2 = contact_detail::point_triangle_dist2(
                                    node_coords[sn],
                                    node_coords[mf.nodes[0]],
                                    node_coords[mf.nodes[2]],
                                    node_coords[mf.nodes[3]],
                                    closest2);
                                if (d2_2 < d2) {
                                    d2 = d2_2;
                                    for (int i=0;i<3;++i) closest[i]=closest2[i];
                                }
                            }

                            if (d2 < best_dist2) {
                                best_dist2 = d2;
                                best_face = fi;
                                for (int i=0;i<3;++i) {
                                    best_closest[i] = closest[i];
                                    best_normal[i] = mf.normal[i];
                                }
                            }
                        }
                    }
                }
            }

            if (best_face >= 0) {
                ContactPair cp;
                cp.slave_node = sn;
                cp.master_face = best_face;
                Real diff[3] = {
                    node_coords[sn][0]-best_closest[0],
                    node_coords[sn][1]-best_closest[1],
                    node_coords[sn][2]-best_closest[2]
                };
                cp.gap = contact_detail::dot3(diff, best_normal);
                for (int i=0;i<3;++i) cp.normal[i] = best_normal[i];
                pairs.push_back(cp);
            }
        }

        return static_cast<int>(pairs.size());
    }

    /**
     * @brief Spatial bucket search for nodes near faces.
     *
     * @param nodes Node coordinates [num_nodes][3]
     * @param num_nodes Number of nodes
     * @param faces Surface faces
     * @param search_radius Search radius
     * @param result Output: for each node, list of nearby face indices
     */
    void bucket_search(const Real nodes[][3], int num_nodes,
                       const std::vector<SurfaceFace>& faces,
                       Real search_radius,
                       std::vector<std::vector<int>>& result) const {
        result.resize(num_nodes);
        for (auto& v : result) v.clear();
        if (faces.empty()) return;

        Real r2 = search_radius * search_radius;

        for (int n = 0; n < num_nodes; ++n) {
            for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
                Real dx = nodes[n][0] - faces[f].centroid[0];
                Real dy = nodes[n][1] - faces[f].centroid[1];
                Real dz = nodes[n][2] - faces[f].centroid[2];
                if (dx*dx + dy*dy + dz*dz < r2)
                    result[n].push_back(f);
            }
        }
    }
};

// ============================================================================
// 21h: Contact2D — Plane Strain and Axisymmetric Contact
// ============================================================================

/**
 * @brief 2D contact for plane strain and axisymmetric problems.
 *
 * Node-to-segment contact in 2D:
 * - Master surface defined by line segments
 * - Slave node projects onto nearest master segment
 * - Gap computed from signed distance to segment with outward normal
 * - Penalty force proportional to penetration depth
 *
 * For axisymmetric problems, the contact force includes the circumferential
 * factor 2*pi*r, where r is the radial coordinate of the contact point.
 */
class Contact2D {
public:
    enum class Mode { PlaneStrain, Axisymmetric };

    /// 2D contact result
    struct ContactResult2D {
        Real gap;
        Real normal[2];
        Real contact_point[2];
        bool in_contact;

        ContactResult2D() : gap(0.0), in_contact(false) {
            normal[0] = normal[1] = 0.0;
            contact_point[0] = contact_point[1] = 0.0;
        }
    };

    Contact2D() = default;
    explicit Contact2D(Mode mode) : mode_(mode) {}

    void set_mode(Mode mode) { mode_ = mode; }
    Mode mode() const { return mode_; }

    /**
     * @brief Detect contact between a slave node and master segments in 2D.
     *
     * Projects the slave node onto each master segment, finds the closest,
     * and computes the signed gap and outward normal.
     *
     * @param slave_node Slave node position [2]
     * @param master_nodes Master segment endpoint coordinates [num_segs+1][2]
     * @param num_segments Number of master segments
     * @param gap Output: signed gap (negative = penetration)
     * @param normal Output: contact normal [2] (outward from master)
     * @return true if potential contact detected (gap < search_tolerance)
     */
    KOKKOS_INLINE_FUNCTION
    bool detect_2d(const Real slave_node[2],
                   const Real master_nodes[][2],
                   int num_segments,
                   Real& gap, Real normal[2]) const {
        Real min_dist2 = 1.0e30;
        Real best_normal[2] = {0.0, 0.0};
        Real best_closest[2] = {0.0, 0.0};

        for (int s = 0; s < num_segments; ++s) {
            const Real* a = master_nodes[s];
            const Real* b = master_nodes[s + 1];

            Real ab[2] = { b[0]-a[0], b[1]-a[1] };
            Real len2 = ab[0]*ab[0] + ab[1]*ab[1];

            Real ap[2] = { slave_node[0]-a[0], slave_node[1]-a[1] };
            Real t = 0.0;
            if (len2 > 1.0e-30) {
                t = (ap[0]*ab[0] + ap[1]*ab[1]) / len2;
                t = std::max(0.0, std::min(1.0, t));
            }

            Real cp[2] = { a[0]+t*ab[0], a[1]+t*ab[1] };
            Real dx = slave_node[0]-cp[0];
            Real dy = slave_node[1]-cp[1];
            Real d2 = dx*dx + dy*dy;

            if (d2 < min_dist2) {
                min_dist2 = d2;
                best_closest[0] = cp[0];
                best_closest[1] = cp[1];

                // Outward normal: rotate segment tangent 90 degrees CCW
                Real len = std::sqrt(len2);
                if (len > 1.0e-30) {
                    best_normal[0] = -ab[1] / len;
                    best_normal[1] =  ab[0] / len;
                }
            }
        }

        Real diff[2] = { slave_node[0]-best_closest[0], slave_node[1]-best_closest[1] };
        gap = diff[0]*best_normal[0] + diff[1]*best_normal[1];
        normal[0] = best_normal[0];
        normal[1] = best_normal[1];

        return (gap < search_tol_);
    }

    /**
     * @brief Compute penalty contact force in 2D.
     *
     * Force = penalty * |gap| * normal  (only when gap < 0)
     *
     * For axisymmetric: F_total = 2 * pi * r * penalty * |gap| * normal
     *
     * @param gap Signed gap (negative = penetration)
     * @param normal Contact normal [2]
     * @param penalty Penalty stiffness
     * @param contact_point Contact point position [2] (x = r for axisymmetric)
     * @param force Output: contact force [2]
     */
    KOKKOS_INLINE_FUNCTION
    void compute_contact_force_2d(Real gap, const Real normal[2],
                                  Real penalty, const Real contact_point[2],
                                  Real force[2]) const {
        force[0] = force[1] = 0.0;
        if (gap >= 0.0) return;

        Real pen_depth = -gap;
        Real f_mag = penalty * pen_depth;

        if (mode_ == Mode::Axisymmetric) {
            Real r = contact_point[0];
            if (r > 1.0e-30) {
                f_mag *= 2.0 * 3.14159265358979323846 * r;
            }
        }

        force[0] = f_mag * normal[0];
        force[1] = f_mag * normal[1];
    }

private:
    Mode mode_ = Mode::PlaneStrain;
    Real search_tol_ = 0.01;
};

// ============================================================================
// 21i: SPHContact — Particle-to-Surface Contact
// ============================================================================

/**
 * @brief SPH particle-to-FEM surface contact.
 *
 * Detects proximity of SPH particles to triangular FEM surface segments.
 * Computes penalty-based contact forces when a particle penetrates the
 * surface (distance < smoothing length).
 *
 * Contact force: f = k_contact * max(0, h - d) * n
 * where d = distance from particle center to surface, h = smoothing length,
 * and n = surface outward normal.
 */
class SPHContact {
public:
    /// SPH-FEM contact pair
    struct SPHContactPair {
        int particle_id;
        int face_id;
        Real gap;           ///< Distance from particle center to surface
        Real normal[3];     ///< Contact normal (pointing away from surface)
        Real contact_pt[3]; ///< Closest point on surface

        SPHContactPair() : particle_id(-1), face_id(-1), gap(0.0) {
            normal[0] = normal[1] = normal[2] = 0.0;
            contact_pt[0] = contact_pt[1] = contact_pt[2] = 0.0;
        }
    };

    SPHContact() = default;

    /**
     * @brief Detect SPH particles in contact with surface faces.
     *
     * For each particle, finds the closest triangular surface face within
     * the search radius. Uses centroid distance as a fast rejection test.
     *
     * @param particle_pos Particle positions [num_particles][3]
     * @param num_particles Number of SPH particles
     * @param face_nodes Face connectivity [num_faces][3]
     * @param node_coords Node coordinates
     * @param num_faces Number of surface faces
     * @param search_radius SPH smoothing length
     * @param contacts Output: detected contacts
     * @return Number of contacts
     */
    int detect_sph_contact(const Real particle_pos[][3],
                           int num_particles,
                           const int face_nodes[][3],
                           const Real node_coords[][3],
                           int num_faces,
                           Real search_radius,
                           std::vector<SPHContactPair>& contacts) const {
        contacts.clear();
        Real r2 = search_radius * search_radius;

        for (int p = 0; p < num_particles; ++p) {
            Real best_dist2 = r2;
            int best_face = -1;
            Real best_closest[3] = {0, 0, 0};

            for (int f = 0; f < num_faces; ++f) {
                // Fast centroid rejection
                Real cx = (node_coords[face_nodes[f][0]][0]
                         + node_coords[face_nodes[f][1]][0]
                         + node_coords[face_nodes[f][2]][0]) / 3.0;
                Real cy = (node_coords[face_nodes[f][0]][1]
                         + node_coords[face_nodes[f][1]][1]
                         + node_coords[face_nodes[f][2]][1]) / 3.0;
                Real cz = (node_coords[face_nodes[f][0]][2]
                         + node_coords[face_nodes[f][1]][2]
                         + node_coords[face_nodes[f][2]][2]) / 3.0;

                Real dx = particle_pos[p][0]-cx;
                Real dy = particle_pos[p][1]-cy;
                Real dz = particle_pos[p][2]-cz;
                if (dx*dx + dy*dy + dz*dz > 4.0*r2) continue;

                Real closest[3];
                Real d2 = contact_detail::point_triangle_dist2(
                    particle_pos[p],
                    node_coords[face_nodes[f][0]],
                    node_coords[face_nodes[f][1]],
                    node_coords[face_nodes[f][2]],
                    closest);

                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_face = f;
                    for (int i=0;i<3;++i) best_closest[i] = closest[i];
                }
            }

            if (best_face >= 0) {
                SPHContactPair cp;
                cp.particle_id = p;
                cp.face_id = best_face;
                cp.gap = std::sqrt(best_dist2);
                for (int i=0;i<3;++i) cp.contact_pt[i] = best_closest[i];

                Real diff[3] = {
                    particle_pos[p][0]-best_closest[0],
                    particle_pos[p][1]-best_closest[1],
                    particle_pos[p][2]-best_closest[2]
                };
                Real dist = contact_detail::norm3(diff);
                if (dist > 1.0e-30) {
                    for (int i=0;i<3;++i) cp.normal[i] = diff[i] / dist;
                } else {
                    Real e0[3], e1[3];
                    contact_detail::sub3(node_coords[face_nodes[best_face][1]],
                                         node_coords[face_nodes[best_face][0]], e0);
                    contact_detail::sub3(node_coords[face_nodes[best_face][2]],
                                         node_coords[face_nodes[best_face][0]], e1);
                    contact_detail::cross3(e0, e1, cp.normal);
                    contact_detail::normalize3(cp.normal);
                }

                contacts.push_back(cp);
            }
        }

        return static_cast<int>(contacts.size());
    }

    /**
     * @brief Compute SPH contact force for a single particle-surface pair.
     *
     * Penalty-based: f = penalty * max(0, smoothing_length - distance) * normal
     *
     * @param particle_pos Particle position [3]
     * @param face_node0, face_node1, face_node2 Triangle vertex positions [3]
     * @param penalty Penalty stiffness (N/m)
     * @param smoothing_length SPH smoothing length (contact radius)
     * @param force Output: contact force on particle [3]
     * @return true if in contact
     */
    KOKKOS_INLINE_FUNCTION
    bool compute_sph_contact_force(const Real particle_pos[3],
                                   const Real face_node0[3],
                                   const Real face_node1[3],
                                   const Real face_node2[3],
                                   Real penalty,
                                   Real smoothing_length,
                                   Real force[3]) const {
        force[0] = force[1] = force[2] = 0.0;

        Real closest[3];
        Real d2 = contact_detail::point_triangle_dist2(
            particle_pos, face_node0, face_node1, face_node2, closest);

        Real dist = std::sqrt(d2);
        Real penetration = smoothing_length - dist;
        if (penetration <= 0.0) return false;

        Real n[3] = {
            particle_pos[0]-closest[0],
            particle_pos[1]-closest[1],
            particle_pos[2]-closest[2]
        };
        if (dist > 1.0e-30) {
            for (int i=0;i<3;++i) n[i] /= dist;
        } else {
            Real e0[3], e1[3];
            contact_detail::sub3(face_node1, face_node0, e0);
            contact_detail::sub3(face_node2, face_node0, e1);
            contact_detail::cross3(e0, e1, n);
            contact_detail::normalize3(n);
        }

        Real f_mag = penalty * penetration;
        for (int i=0;i<3;++i) force[i] = f_mag * n[i];

        return true;
    }
};

// ============================================================================
// 21j: AirbagFabricContact — Fabric Self-Contact with Porosity
// ============================================================================

/**
 * @brief Airbag fabric self-contact with porosity effects.
 *
 * Models self-intersection detection for folded fabric (e.g. airbag deployment).
 * Includes:
 * - Self-contact detection using proximity search on triangulated fabric mesh
 * - Contact force based on penalty and pressure loading
 * - Leak-through detection and flow rate computation (Wang-Nefske model)
 *
 * Leak rate: Q = C_p * A * sqrt(2 * dP / rho)
 */
class AirbagFabricContact {
public:
    /// Self-contact pair: node vs. non-adjacent triangle
    struct FabricContactPair {
        int tri_a;        ///< Triangle containing the node
        int tri_b;        ///< Opposing triangle
        int node_id;      ///< Node in contact
        Real gap;         ///< Gap distance (negative = penetration through thickness)
        Real normal[3];   ///< Contact normal

        FabricContactPair() : tri_a(-1), tri_b(-1), node_id(-1), gap(0.0) {
            normal[0] = normal[1] = normal[2] = 0.0;
        }
    };

    AirbagFabricContact() = default;

    void set_thickness(Real t) { thickness_ = t; }
    void set_porosity_coefficient(Real cp) { porosity_coeff_ = cp; }

    /**
     * @brief Detect self-contact in a triangulated fabric mesh.
     *
     * For each node, checks against all non-adjacent triangles within
     * a search radius (2x fabric thickness). Adjacent triangles (sharing
     * a node) are excluded.
     *
     * @param node_coords Node coordinates [num_nodes][3]
     * @param num_nodes Number of nodes
     * @param triangles Triangle connectivity [num_tris][3]
     * @param num_tris Number of triangles
     * @param thickness Fabric thickness
     * @param contacts Output: detected self-contact pairs
     * @return Number of contacts
     */
    int detect_self_contact(const Real node_coords[][3],
                            int num_nodes,
                            const int triangles[][3],
                            int num_tris,
                            Real thickness,
                            std::vector<FabricContactPair>& contacts) const {
        contacts.clear();
        Real search_r = 2.0 * thickness;
        Real search_r2 = search_r * search_r;

        // Build node-to-triangle adjacency
        std::vector<std::vector<int>> node_tris(num_nodes);
        for (int t = 0; t < num_tris; ++t) {
            for (int i = 0; i < 3; ++i) {
                int n = triangles[t][i];
                if (n >= 0 && n < num_nodes)
                    node_tris[n].push_back(t);
            }
        }

        // Triangle centroids for fast rejection
        std::vector<std::array<Real, 3>> tri_cen(num_tris);
        for (int t = 0; t < num_tris; ++t) {
            tri_cen[t] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i) {
                int n = triangles[t][i];
                tri_cen[t][0] += node_coords[n][0];
                tri_cen[t][1] += node_coords[n][1];
                tri_cen[t][2] += node_coords[n][2];
            }
            for (int d=0;d<3;++d) tri_cen[t][d] /= 3.0;
        }

        for (int n = 0; n < num_nodes; ++n) {
            std::unordered_set<int> adj_tris(node_tris[n].begin(), node_tris[n].end());

            for (int t = 0; t < num_tris; ++t) {
                if (adj_tris.count(t)) continue;

                Real dx = node_coords[n][0]-tri_cen[t][0];
                Real dy = node_coords[n][1]-tri_cen[t][1];
                Real dz = node_coords[n][2]-tri_cen[t][2];
                if (dx*dx+dy*dy+dz*dz > 4.0*search_r2) continue;

                Real closest[3];
                Real d2 = contact_detail::point_triangle_dist2(
                    node_coords[n],
                    node_coords[triangles[t][0]],
                    node_coords[triangles[t][1]],
                    node_coords[triangles[t][2]],
                    closest);

                if (d2 < search_r2) {
                    FabricContactPair cp;
                    cp.node_id = n;
                    cp.tri_a = node_tris[n].empty() ? -1 : node_tris[n][0];
                    cp.tri_b = t;
                    cp.gap = std::sqrt(d2) - thickness;

                    Real diff[3] = {
                        node_coords[n][0]-closest[0],
                        node_coords[n][1]-closest[1],
                        node_coords[n][2]-closest[2]
                    };
                    Real dist = std::sqrt(d2);
                    if (dist > 1.0e-30) {
                        for (int i=0;i<3;++i) cp.normal[i] = diff[i] / dist;
                    }

                    contacts.push_back(cp);
                }
            }
        }

        return static_cast<int>(contacts.size());
    }

    /**
     * @brief Compute fabric contact force.
     *
     * Combines penalty-based gap closure and pressure effects:
     *   f_total = penalty * max(0, -gap) * area + pressure * area * proximity_factor
     *
     * @param gap Signed gap (negative = penetration through thickness)
     * @param area Tributary contact area
     * @param pressure Internal airbag pressure
     * @param penalty Penalty stiffness
     * @param force Output: scalar contact force magnitude
     */
    KOKKOS_INLINE_FUNCTION
    void compute_fabric_contact_force(Real gap, Real area,
                                      Real pressure, Real penalty,
                                      Real& force) const {
        force = 0.0;

        if (gap < 0.0) {
            force += penalty * (-gap) * area;
        }

        if (gap < thickness_ && gap > -thickness_) {
            Real proximity_factor = 1.0 - std::abs(gap) / thickness_;
            force += pressure * area * proximity_factor;
        }
    }

    /**
     * @brief Compute gas leak rate through fabric mesh.
     *
     * Wang-Nefske porosity model:
     *   Q = C_p(h) * A * sqrt(2 * |dP| / rho)
     *
     * C_p depends on mesh spacing: C_p = base_porosity * (h / h_ref)^2
     *
     * @param mesh_spacing Average element edge length
     * @param pressure_diff Pressure difference across fabric (Pa)
     * @param gas_density Gas density (kg/m^3)
     * @param area Fabric area (m^2)
     * @return Volumetric leak rate (m^3/s)
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_leak_rate(Real mesh_spacing, Real pressure_diff,
                           Real gas_density = 1.2, Real area = 1.0) const {
        if (pressure_diff <= 0.0 || gas_density < 1.0e-30) return 0.0;

        Real h_ref = 0.005; // 5mm reference spacing
        Real h_ratio = mesh_spacing / h_ref;
        Real Cp = porosity_coeff_ * h_ratio * h_ratio;
        if (Cp > 1.0) Cp = 1.0;

        return Cp * area * std::sqrt(2.0 * pressure_diff / gas_density);
    }

private:
    Real thickness_ = 0.001;
    Real porosity_coeff_ = 0.001;
};

// ============================================================================
// 21k: ContactHeatGeneration — Friction-Induced Heat
// ============================================================================

/**
 * @brief Friction-induced heat generation at contact interfaces.
 *
 * Heat flux: q = eta * mu * p * v_slip
 * where eta = fraction of frictional work converted to heat.
 *
 * Heat partition (Charron 1943):
 *   f = sqrt(k1*rho1*cp1) / (sqrt(k1*rho1*cp1) + sqrt(k2*rho2*cp2))
 *   q_master = f * q,  q_slave = (1-f) * q
 *
 * Reference:
 * - Wriggers (2006) Ch. 12 "Thermo-mechanical contact"
 */
class ContactHeatGeneration {
public:
    /// Thermal properties for heat partition calculation
    struct ThermalProps {
        Real conductivity;
        Real density;
        Real specific_heat;

        ThermalProps() : conductivity(50.0), density(7800.0), specific_heat(500.0) {}
        ThermalProps(Real k, Real rho, Real cp)
            : conductivity(k), density(rho), specific_heat(cp) {}

        /// Thermal effusivity: sqrt(k * rho * cp)
        KOKKOS_INLINE_FUNCTION
        Real effusivity() const {
            return std::sqrt(conductivity * density * specific_heat);
        }
    };

    ContactHeatGeneration() = default;

    void set_heat_fraction(Real eta) { eta_ = eta; }
    void set_master_properties(const ThermalProps& props) { master_props_ = props; }
    void set_slave_properties(const ThermalProps& props) { slave_props_ = props; }

    /**
     * @brief Compute frictional heat flux.
     *
     * q = eta * mu * |pressure| * |v_slip|
     *
     * @param friction_coeff Friction coefficient
     * @param pressure Contact pressure (Pa)
     * @param slip_velocity Tangential slip velocity (m/s)
     * @param partition_factor Heat fraction to master (0 to 1)
     * @return Total heat flux (W/m^2)
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_heat_flux(Real friction_coeff, Real pressure,
                           Real slip_velocity, Real partition_factor) const {
        return eta_ * friction_coeff * std::abs(pressure) * std::abs(slip_velocity);
    }

    /**
     * @brief Compute Charron partition factor from material properties.
     *
     * f = e_master / (e_master + e_slave)
     *
     * @return Partition factor for master surface [0, 1]
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_partition_factor() const {
        Real e_m = master_props_.effusivity();
        Real e_s = slave_props_.effusivity();
        Real sum = e_m + e_s;
        if (sum < 1.0e-30) return 0.5;
        return e_m / sum;
    }

    /**
     * @brief Distribute frictional heat to master and slave node temperatures.
     *
     * For each contact pair:
     *   q_total = eta * mu * p * v_slip
     *   dT_master = f * q * A * dt / (m_master * cp_master)
     *   dT_slave = (1-f) * q * A * dt / (m_slave * cp_slave)
     *
     * @param num_pairs Number of contact pairs
     * @param friction_coeff Friction coefficient
     * @param pressures Contact pressures [num_pairs]
     * @param slip_velocities Slip velocities [num_pairs]
     * @param tributary_areas Contact areas [num_pairs]
     * @param node_temperatures Node temperatures (modified) [num_nodes]
     * @param node_masses Lumped masses [num_nodes]
     * @param dt Time step
     * @param master_node_ids Master node per pair [num_pairs]
     * @param slave_node_ids Slave node per pair [num_pairs]
     */
    void distribute_heat(int num_pairs,
                         Real friction_coeff,
                         const Real* pressures,
                         const Real* slip_velocities,
                         const Real* tributary_areas,
                         Real* node_temperatures,
                         const Real* node_masses,
                         Real dt,
                         const int* master_node_ids,
                         const int* slave_node_ids) const {
        Real f = compute_partition_factor();

        for (int i = 0; i < num_pairs; ++i) {
            Real q = compute_heat_flux(friction_coeff, pressures[i],
                                       slip_velocities[i], f);
            Real Q_total = q * tributary_areas[i];

            int mn = master_node_ids[i];
            if (mn >= 0 && node_masses[mn] > 1.0e-30) {
                Real dT = f * Q_total * dt / (node_masses[mn] * master_props_.specific_heat);
                node_temperatures[mn] += dT;
            }

            int sn = slave_node_ids[i];
            if (sn >= 0 && node_masses[sn] > 1.0e-30) {
                Real dT = (1.0-f) * Q_total * dt / (node_masses[sn] * slave_props_.specific_heat);
                node_temperatures[sn] += dT;
            }
        }
    }

private:
    Real eta_ = 0.95;
    ThermalProps master_props_;
    ThermalProps slave_props_;
};

// ============================================================================
// 21l: MortarContactFriction — Mortar Contact with Coulomb Friction
// ============================================================================

/**
 * @brief Production-grade mortar contact with Coulomb friction.
 *
 * Mortar method enforces contact constraints via weighted integrals:
 *   g_w = integral_{Gamma_c} N_slave * (x_slave - x_master) . n dA
 *
 * KKT conditions: lambda >= 0, g_w >= 0, lambda * g_w = 0
 * Coulomb friction: |t_tangent| <= mu * lambda
 *
 * Penalty regularization:
 *   lambda = epsilon_N * max(0, -g_w)
 *
 * Consistent linearization for Newton-Raphson convergence.
 *
 * References:
 * - Puso & Laursen (2004) "A mortar segment-to-segment contact method"
 * - Popp et al. (2010) "A dual mortar approach for 3D finite deformation contact"
 */
class MortarContactFriction {
public:
    /// Mortar integration point data
    struct MortarIntegral {
        Real weight;           ///< Quadrature weight * Jacobian
        Real shape_master[4];  ///< Master shape functions
        Real shape_slave[4];   ///< Slave shape functions
        Real normal[3];        ///< Contact normal
        Real xi_master[2];     ///< Master parametric coordinates
        Real xi_slave[2];      ///< Slave parametric coordinates

        MortarIntegral() : weight(0.0) {
            for (int i = 0; i < 4; ++i) { shape_master[i] = 0.0; shape_slave[i] = 0.0; }
            normal[0] = normal[1] = normal[2] = 0.0;
            xi_master[0] = xi_master[1] = 0.0;
            xi_slave[0] = xi_slave[1] = 0.0;
        }
    };

    /// Mortar contact state
    struct MortarState {
        Real weighted_gap;
        Real lambda_n;
        Real lambda_t[2];
        Real slip[2];
        bool active;
        bool sliding;

        MortarState() : weighted_gap(0.0), lambda_n(0.0), active(false), sliding(false) {
            lambda_t[0] = lambda_t[1] = 0.0;
            slip[0] = slip[1] = 0.0;
        }
    };

    MortarContactFriction() = default;

    void set_normal_penalty(Real eps_n) { epsilon_n_ = eps_n; }
    void set_tangential_penalty(Real eps_t) { epsilon_t_ = eps_t; }
    void set_friction_coefficient(Real mu) { mu_ = mu; }

    /**
     * @brief Compute mortar integrals for a master-slave segment pair.
     *
     * Uses 2x2 Gauss quadrature. For each quadrature point, evaluates
     * bilinear shape functions on both sides and computes weighted integration.
     *
     * @param master_coords Master segment node coordinates [4][3]
     * @param slave_coords Slave segment node coordinates [4][3]
     * @param num_master_nodes 3 or 4
     * @param num_slave_nodes 3 or 4
     * @param integrals Output: integration point data
     * @param num_integrals Output: number of integration points
     */
    void compute_mortar_integrals(const Real master_coords[][3],
                                  const Real slave_coords[][3],
                                  int num_master_nodes,
                                  int num_slave_nodes,
                                  MortarIntegral* integrals,
                                  int& num_integrals) const {
        static const Real gp = 1.0 / std::sqrt(3.0);
        static const Real gauss_pts[4][2] = {
            {-gp, -gp}, {gp, -gp}, {-gp, gp}, {gp, gp}
        };
        static const Real gauss_wts[4] = { 1.0, 1.0, 1.0, 1.0 };

        num_integrals = 4;

        // Slave surface normal
        Real slave_normal[3] = {0, 0, 0};
        {
            Real e0[3], e1[3];
            contact_detail::sub3(slave_coords[1], slave_coords[0], e0);
            int idx = (num_slave_nodes >= 3) ? 2 : 1;
            contact_detail::sub3(slave_coords[idx], slave_coords[0], e1);
            contact_detail::cross3(e0, e1, slave_normal);
            contact_detail::normalize3(slave_normal);
        }

        for (int g = 0; g < 4; ++g) {
            Real xi = gauss_pts[g][0];
            Real eta = gauss_pts[g][1];

            MortarIntegral& mi = integrals[g];
            mi.xi_slave[0] = xi;
            mi.xi_slave[1] = eta;
            for (int i=0;i<3;++i) mi.normal[i] = slave_normal[i];

            // Slave shape functions (bilinear quad or linear tri)
            if (num_slave_nodes == 4) {
                mi.shape_slave[0] = 0.25 * (1.0-xi) * (1.0-eta);
                mi.shape_slave[1] = 0.25 * (1.0+xi) * (1.0-eta);
                mi.shape_slave[2] = 0.25 * (1.0+xi) * (1.0+eta);
                mi.shape_slave[3] = 0.25 * (1.0-xi) * (1.0+eta);
            } else {
                Real L1 = 0.5*(1.0-xi), L2 = 0.5*(1.0+xi), L3 = 0.5*(1.0+eta);
                Real sum = L1+L2+L3;
                if (sum > 1.0e-30) { L1/=sum; L2/=sum; L3/=sum; }
                mi.shape_slave[0] = L1;
                mi.shape_slave[1] = L2;
                mi.shape_slave[2] = L3;
                mi.shape_slave[3] = 0.0;
            }

            // Physical position on slave surface
            Real x_slave[3] = {0, 0, 0};
            for (int i = 0; i < num_slave_nodes; ++i) {
                for (int d=0;d<3;++d)
                    x_slave[d] += mi.shape_slave[i] * slave_coords[i][d];
            }

            // Project onto master (inverse mapping via Newton)
            Real xi_m = 0.0, eta_m = 0.0;
            project_to_master(x_slave, master_coords, num_master_nodes, xi_m, eta_m);
            mi.xi_master[0] = xi_m;
            mi.xi_master[1] = eta_m;

            // Master shape functions
            if (num_master_nodes == 4) {
                mi.shape_master[0] = 0.25 * (1.0-xi_m) * (1.0-eta_m);
                mi.shape_master[1] = 0.25 * (1.0+xi_m) * (1.0-eta_m);
                mi.shape_master[2] = 0.25 * (1.0+xi_m) * (1.0+eta_m);
                mi.shape_master[3] = 0.25 * (1.0-xi_m) * (1.0+eta_m);
            } else {
                Real L1 = 0.5*(1.0-xi_m), L2 = 0.5*(1.0+xi_m), L3 = 0.5*(1.0+eta_m);
                Real sum = L1+L2+L3;
                if (sum > 1.0e-30) { L1/=sum; L2/=sum; L3/=sum; }
                mi.shape_master[0] = L1;
                mi.shape_master[1] = L2;
                mi.shape_master[2] = L3;
                mi.shape_master[3] = 0.0;
            }

            // Jacobian determinant for slave surface area
            Real J = compute_slave_jacobian(slave_coords, num_slave_nodes, xi, eta);
            mi.weight = gauss_wts[g] * J;
        }
    }

    /**
     * @brief Compute mortar-weighted gap from integrals and displacements.
     *
     * g_w = sum_gp { w_gp * [N_slave*x_slave - N_master*x_master] . n }
     *
     * @param integrals Mortar integration points
     * @param num_integrals Number of integration points
     * @param slave_disp Slave displacements [4][3]
     * @param master_disp Master displacements [4][3]
     * @param slave_coords Slave reference coords [4][3]
     * @param master_coords Master reference coords [4][3]
     * @param num_slave Number of slave nodes
     * @param num_master Number of master nodes
     * @return Weighted gap (negative = penetration)
     */
    Real compute_weighted_gap(const MortarIntegral* integrals,
                              int num_integrals,
                              const Real slave_disp[][3],
                              const Real master_disp[][3],
                              const Real slave_coords[][3],
                              const Real master_coords[][3],
                              int num_slave, int num_master) const {
        Real g_w = 0.0;

        for (int g = 0; g < num_integrals; ++g) {
            const MortarIntegral& mi = integrals[g];

            Real x_s[3] = {0, 0, 0};
            for (int i = 0; i < num_slave; ++i) {
                for (int d=0;d<3;++d)
                    x_s[d] += mi.shape_slave[i] * (slave_coords[i][d] + slave_disp[i][d]);
            }

            Real x_m[3] = {0, 0, 0};
            for (int i = 0; i < num_master; ++i) {
                for (int d=0;d<3;++d)
                    x_m[d] += mi.shape_master[i] * (master_coords[i][d] + master_disp[i][d]);
            }

            Real gap_vec[3] = { x_s[0]-x_m[0], x_s[1]-x_m[1], x_s[2]-x_m[2] };
            g_w += mi.weight * contact_detail::dot3(gap_vec, mi.normal);
        }

        return g_w;
    }

    /**
     * @brief Compute contact residual for Newton-Raphson assembly.
     *
     * Normal:
     *   R_n = epsilon_n * max(0, -gap)  if gap < 0
     *   R_n = 0                          if gap >= 0
     *
     * Tangential (Coulomb friction):
     *   trial_t = epsilon_t * slip
     *   if |trial_t| <= mu * R_n: R_t = trial_t           (stick)
     *   else:                     R_t = mu * R_n * t/|t|  (slip)
     *
     * @param gap Mortar-weighted gap
     * @param lambda_n Current normal pressure
     * @param friction_coeff Friction coefficient
     * @param slip Tangential slip [2]
     * @param R_n Output: normal residual
     * @param R_t Output: tangential residual [2]
     * @param is_sliding Output: true if sliding
     */
    KOKKOS_INLINE_FUNCTION
    void contact_residual(Real gap, Real lambda_n,
                          Real friction_coeff,
                          const Real slip[2],
                          Real& R_n, Real R_t[2],
                          bool& is_sliding) const {
        if (gap < 0.0) {
            R_n = epsilon_n_ * (-gap);
        } else {
            R_n = 0.0;
            R_t[0] = R_t[1] = 0.0;
            is_sliding = false;
            return;
        }

        Real trial_t[2] = { epsilon_t_ * slip[0], epsilon_t_ * slip[1] };
        Real trial_mag = std::sqrt(trial_t[0]*trial_t[0] + trial_t[1]*trial_t[1]);
        Real friction_limit = friction_coeff * R_n;

        if (trial_mag <= friction_limit) {
            R_t[0] = trial_t[0];
            R_t[1] = trial_t[1];
            is_sliding = false;
        } else {
            if (trial_mag > 1.0e-30) {
                R_t[0] = friction_limit * trial_t[0] / trial_mag;
                R_t[1] = friction_limit * trial_t[1] / trial_mag;
            } else {
                R_t[0] = R_t[1] = 0.0;
            }
            is_sliding = true;
        }
    }

    /**
     * @brief Compute contact tangent stiffness for Newton linearization.
     *
     * Returns the 3x3 tangent K_c: dF_c = K_c * du
     *
     * Normal: K_nn = epsilon_n * n (x) n  (if active)
     * Tangential: K_tt = epsilon_t * I_t  (stick)
     *             K_tt = mu * epsilon_n * P_t  (slip)
     *
     * @param gap Current gap
     * @param slip Tangential slip [2]
     * @param normal Contact normal [3]
     * @param tangent Output: 3x3 tangent matrix (row-major)
     */
    void contact_tangent(Real gap, const Real slip[2],
                         const Real normal[3],
                         Real tangent[9]) const {
        for (int i = 0; i < 9; ++i) tangent[i] = 0.0;
        if (gap >= 0.0) return;

        // Normal contribution: K_nn = epsilon_n * n (x) n
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                tangent[i*3+j] += epsilon_n_ * normal[i] * normal[j];

        // Build tangent plane basis
        Real t1[3], t2[3];
        build_tangent_basis(normal, t1, t2);

        Real slip_mag = std::sqrt(slip[0]*slip[0] + slip[1]*slip[1]);
        Real friction_limit = mu_ * epsilon_n_ * (-gap);
        Real trial_mag = epsilon_t_ * slip_mag;

        if (trial_mag <= friction_limit) {
            // Stick: K_tt = epsilon_t * (t1(x)t1 + t2(x)t2)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    tangent[i*3+j] += epsilon_t_ * (t1[i]*t1[j] + t2[i]*t2[j]);
        } else {
            // Slip: K_tt = mu * epsilon_n * projection
            Real kt = mu_ * epsilon_n_;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    tangent[i*3+j] += kt * (t1[i]*t1[j] + t2[i]*t2[j]);

            // Rank-1 correction for slip direction change
            if (slip_mag > 1.0e-30) {
                Real s3d[3] = {
                    slip[0]*t1[0] + slip[1]*t2[0],
                    slip[0]*t1[1] + slip[1]*t2[1],
                    slip[0]*t1[2] + slip[1]*t2[2]
                };
                Real s_mag = contact_detail::norm3(s3d);
                if (s_mag > 1.0e-30) {
                    Real inv_s2 = 1.0 / (s_mag * s_mag);
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            tangent[i*3+j] -= kt * s3d[i] * s3d[j] * inv_s2;
                }
            }
        }
    }

private:
    Real epsilon_n_ = 1.0e10;
    Real epsilon_t_ = 1.0e10;
    Real mu_ = 0.3;

    /// Project point onto master segment via Newton-Raphson inverse mapping
    void project_to_master(const Real target[3],
                           const Real master_coords[][3],
                           int num_nodes,
                           Real& xi, Real& eta) const {
        xi = 0.0; eta = 0.0;

        for (int iter = 0; iter < 10; ++iter) {
            Real x[3] = {0, 0, 0};
            Real dx_dxi[3] = {0, 0, 0};
            Real dx_deta[3] = {0, 0, 0};

            if (num_nodes == 4) {
                Real N[4] = {
                    0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta),
                    0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta)
                };
                Real dN_dxi[4] = {
                    -0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)
                };
                Real dN_deta[4] = {
                    -0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)
                };
                for (int i = 0; i < 4; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        x[d]       += N[i] * master_coords[i][d];
                        dx_dxi[d]  += dN_dxi[i] * master_coords[i][d];
                        dx_deta[d] += dN_deta[i] * master_coords[i][d];
                    }
                }
            } else {
                for (int d = 0; d < 3; ++d) {
                    x[d] = master_coords[0][d]*(1-xi-eta) + master_coords[1][d]*xi + master_coords[2][d]*eta;
                    dx_dxi[d]  = master_coords[1][d] - master_coords[0][d];
                    dx_deta[d] = master_coords[2][d] - master_coords[0][d];
                }
            }

            Real r[3] = { x[0]-target[0], x[1]-target[1], x[2]-target[2] };
            Real r_xi  = contact_detail::dot3(r, dx_dxi);
            Real r_eta = contact_detail::dot3(r, dx_deta);

            Real J11 = contact_detail::dot3(dx_dxi, dx_dxi);
            Real J12 = contact_detail::dot3(dx_dxi, dx_deta);
            Real J22 = contact_detail::dot3(dx_deta, dx_deta);

            Real det = J11*J22 - J12*J12;
            if (std::abs(det) < 1.0e-30) break;
            Real inv_det = 1.0 / det;

            Real dxi  = -inv_det * (J22*r_xi - J12*r_eta);
            Real deta = -inv_det * (-J12*r_xi + J11*r_eta);

            xi += dxi;
            eta += deta;

            if (std::abs(dxi) + std::abs(deta) < 1.0e-12) break;
        }

        xi  = std::max(-1.0, std::min(1.0, xi));
        eta = std::max(-1.0, std::min(1.0, eta));
    }

    /// Compute slave surface Jacobian determinant at (xi, eta)
    Real compute_slave_jacobian(const Real slave_coords[][3],
                                int num_nodes,
                                Real xi, Real eta) const {
        Real dx_dxi[3] = {0, 0, 0};
        Real dx_deta[3] = {0, 0, 0};

        if (num_nodes == 4) {
            Real dN_dxi[4] = { -0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta) };
            Real dN_deta[4] = { -0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi) };
            for (int i = 0; i < 4; ++i) {
                for (int d = 0; d < 3; ++d) {
                    dx_dxi[d]  += dN_dxi[i] * slave_coords[i][d];
                    dx_deta[d] += dN_deta[i] * slave_coords[i][d];
                }
            }
        } else {
            for (int d = 0; d < 3; ++d) {
                dx_dxi[d]  = slave_coords[1][d] - slave_coords[0][d];
                dx_deta[d] = slave_coords[2][d] - slave_coords[0][d];
            }
        }

        Real cross[3];
        contact_detail::cross3(dx_dxi, dx_deta, cross);
        return contact_detail::norm3(cross);
    }

    /// Build orthonormal tangent basis from a normal vector
    static void build_tangent_basis(const Real n[3], Real t1[3], Real t2[3]) {
        Real abs_n[3] = { std::abs(n[0]), std::abs(n[1]), std::abs(n[2]) };
        Real seed[3] = {0, 0, 0};
        if (abs_n[0] <= abs_n[1] && abs_n[0] <= abs_n[2])
            seed[0] = 1.0;
        else if (abs_n[1] <= abs_n[2])
            seed[1] = 1.0;
        else
            seed[2] = 1.0;

        Real dot_sn = seed[0]*n[0] + seed[1]*n[1] + seed[2]*n[2];
        t1[0] = seed[0] - dot_sn*n[0];
        t1[1] = seed[1] - dot_sn*n[1];
        t1[2] = seed[2] - dot_sn*n[2];
        contact_detail::normalize3(t1);

        contact_detail::cross3(n, t1, t2);
        contact_detail::normalize3(t2);
    }
};

} // namespace fem
} // namespace nxs
