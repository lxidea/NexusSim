#pragma once

/**
 * @file preprocess_wave22.hpp
 * @brief Wave 22: Pre-processing utilities for mesh quality, repair, contact, assignment, and transforms
 *
 * Components:
 * 1. MeshQualityMetrics   - Element quality (aspect ratio, Jacobian ratio, warpage, skewness)
 * 2. MeshRepair           - Duplicate node merging, free edge detection, T-junction detection
 * 3. AutoContactSurface   - Exterior face extraction, part grouping, contact pair creation
 * 4. PartMaterialAssigner - Material assignment to parts, region-based assignment (box/sphere/cylinder)
 * 5. CoordinateTransform  - Rectangular/cylindrical/spherical transforms for nodes, vectors, tensors
 *
 * References:
 * - Knupp (2000) "Achieving Finite Element Mesh Quality via Optimization"
 * - Lo (2015) "Finite Element Mesh Generation"
 * - Frey & George (2000) "Mesh Generation"
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// 1. MeshQualityMetrics
// ============================================================================

/**
 * @brief Quality metrics returned by element quality computations
 */
struct QualityMetrics {
    Real aspect_ratio;      ///< max edge / min edge (1.0 = ideal)
    Real jacobian_ratio;    ///< min |J| / max |J| at corner evaluation points
    Real warpage;           ///< max face out-of-plane angle in degrees (0 = flat)
    Real skewness;          ///< 0 = ideal, 1 = degenerate
    Real min_angle;         ///< minimum interior angle (degrees)
    Real max_angle;         ///< maximum interior angle (degrees)
    bool is_valid;          ///< true if all Jacobians are positive

    QualityMetrics()
        : aspect_ratio(1.0), jacobian_ratio(1.0), warpage(0.0)
        , skewness(0.0), min_angle(90.0), max_angle(90.0), is_valid(true) {}
};

/**
 * @brief Mesh quality evaluation for hexahedral, tetrahedral, and quad shell elements
 *
 * Computes standard quality measures:
 * - Aspect ratio: max edge length / min edge length
 * - Jacobian ratio: min |J| / max |J| evaluated at element corners
 * - Warpage: for shells/hex faces, the angle between split-triangle normals
 * - Skewness: deviation from ideal element shape
 */
class MeshQualityMetrics {
public:
    /**
     * @brief Compute quality for an 8-node hexahedral element
     * @param coords coords[i][j] where i=node(0..7), j=component(0=x,1=y,2=z)
     * @return QualityMetrics struct
     */
    static QualityMetrics compute_hex_quality(const Real coords[8][3]) {
        QualityMetrics q;

        // 12 hex edges
        static const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},   // bottom
            {4,5},{5,6},{6,7},{7,4},   // top
            {0,4},{1,5},{2,6},{3,7}    // vertical
        };

        Real min_e = std::numeric_limits<Real>::max();
        Real max_e = 0.0;
        for (const auto& e : edges) {
            Real len = edge_len(coords[e[0]], coords[e[1]]);
            min_e = std::min(min_e, len);
            max_e = std::max(max_e, len);
        }
        q.aspect_ratio = (min_e > 1e-30) ? max_e / min_e : 1e30;

        // Evaluate Jacobian at all 8 corners
        // At each corner, J = a . (b x c) where a,b,c are the three edge vectors
        // Corner i: edges to the 3 adjacent nodes
        static const int corner_adj[8][3] = {
            {1,3,4}, {2,0,5}, {3,1,6}, {0,2,7},
            {5,7,0}, {6,4,1}, {7,5,2}, {4,6,3}
        };

        Real jac_min = std::numeric_limits<Real>::max();
        Real jac_max = -std::numeric_limits<Real>::max();
        bool all_positive = true;

        for (int c = 0; c < 8; ++c) {
            int n0 = c;
            int n1 = corner_adj[c][0];
            int n2 = corner_adj[c][1];
            int n3 = corner_adj[c][2];

            Real ax = coords[n1][0] - coords[n0][0];
            Real ay = coords[n1][1] - coords[n0][1];
            Real az = coords[n1][2] - coords[n0][2];

            Real bx = coords[n2][0] - coords[n0][0];
            Real by = coords[n2][1] - coords[n0][1];
            Real bz = coords[n2][2] - coords[n0][2];

            Real cx = coords[n3][0] - coords[n0][0];
            Real cy = coords[n3][1] - coords[n0][1];
            Real cz = coords[n3][2] - coords[n0][2];

            Real jac = ax * (by * cz - bz * cy)
                     - ay * (bx * cz - bz * cx)
                     + az * (bx * cy - by * cx);

            if (jac <= 0.0) all_positive = false;
            Real aj = std::abs(jac);
            jac_min = std::min(jac_min, aj);
            jac_max = std::max(jac_max, aj);
        }

        q.jacobian_ratio = (jac_max > 1e-30) ? jac_min / jac_max : 0.0;
        q.is_valid = all_positive;

        // Skewness: deviation of space diagonals from equal length
        Real d1 = pt_dist(coords[0], coords[6]);
        Real d2 = pt_dist(coords[1], coords[7]);
        Real d3 = pt_dist(coords[2], coords[4]);
        Real d4 = pt_dist(coords[3], coords[5]);
        Real avg_d = (d1 + d2 + d3 + d4) / 4.0;
        Real max_dev = 0.0;
        for (Real d : {d1, d2, d3, d4}) {
            max_dev = std::max(max_dev, std::abs(d - avg_d));
        }
        q.skewness = (avg_d > 1e-30) ? max_dev / avg_d : 0.0;

        // Warpage: max warpage across all 6 faces
        static const int faces[6][4] = {
            {0,1,2,3}, {4,7,6,5}, {0,4,5,1},
            {2,6,7,3}, {0,3,7,4}, {1,5,6,2}
        };
        Real max_warp = 0.0;
        for (const auto& f : faces) {
            Real w = face_warpage(coords[f[0]], coords[f[1]], coords[f[2]], coords[f[3]]);
            max_warp = std::max(max_warp, w);
        }
        q.warpage = max_warp;

        // Min/max angles on bottom face
        q.min_angle = 360.0;
        q.max_angle = 0.0;
        for (int i = 0; i < 4; ++i) {
            int p = faces[0][(i + 3) % 4];
            int v = faces[0][i];
            int n = faces[0][(i + 1) % 4];
            Real ang = vertex_angle(coords[p], coords[v], coords[n]);
            q.min_angle = std::min(q.min_angle, ang);
            q.max_angle = std::max(q.max_angle, ang);
        }

        return q;
    }

    /**
     * @brief Compute quality for a 4-node tetrahedral element
     * @param coords coords[i][j] where i=node(0..3), j=component(0=x,1=y,2=z)
     * @return QualityMetrics struct
     */
    static QualityMetrics compute_tet_quality(const Real coords[4][3]) {
        QualityMetrics q;

        // 6 edges
        static const int edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        Real min_e = std::numeric_limits<Real>::max();
        Real max_e = 0.0;
        for (const auto& e : edges) {
            Real len = edge_len(coords[e[0]], coords[e[1]]);
            min_e = std::min(min_e, len);
            max_e = std::max(max_e, len);
        }
        q.aspect_ratio = (min_e > 1e-30) ? max_e / min_e : 1e30;

        // Volume via triple product: V = (1/6) * (a . (b x c))
        Real ax = coords[1][0] - coords[0][0];
        Real ay = coords[1][1] - coords[0][1];
        Real az = coords[1][2] - coords[0][2];
        Real bx = coords[2][0] - coords[0][0];
        Real by = coords[2][1] - coords[0][1];
        Real bz = coords[2][2] - coords[0][2];
        Real cx = coords[3][0] - coords[0][0];
        Real cy = coords[3][1] - coords[0][1];
        Real cz = coords[3][2] - coords[0][2];

        Real vol6 = ax * (by * cz - bz * cy)
                  - ay * (bx * cz - bz * cx)
                  + az * (bx * cy - by * cx);

        q.is_valid = (vol6 > 0.0);
        q.jacobian_ratio = q.is_valid ? 1.0 : 0.0;

        // Tet skewness: compare volume to volume of equilateral tet with same longest edge
        // Equilateral tet volume: V_eq = L^3 / (6*sqrt(2))
        Real v_eq = max_e * max_e * max_e / (6.0 * std::sqrt(2.0));
        Real v_actual = std::abs(vol6) / 6.0;
        q.skewness = (v_eq > 1e-30) ? 1.0 - v_actual / v_eq : 1.0;
        q.skewness = std::max(0.0, std::min(1.0, q.skewness));

        // Face angles on face 0-1-2
        q.min_angle = 360.0;
        q.max_angle = 0.0;
        // Check all 4 faces (6 unique dihedral angles along edges, but simpler: face angles)
        static const int tri_faces[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
        for (const auto& f : tri_faces) {
            for (int k = 0; k < 3; ++k) {
                int prev = f[(k + 2) % 3];
                int vert = f[k];
                int next = f[(k + 1) % 3];
                Real ang = vertex_angle(coords[prev], coords[vert], coords[next]);
                q.min_angle = std::min(q.min_angle, ang);
                q.max_angle = std::max(q.max_angle, ang);
            }
        }

        q.warpage = 0.0; // Tets have no warpage (all faces are planar triangles)
        return q;
    }

    /**
     * @brief Compute quality for a 4-node quadrilateral shell element
     * @param coords coords[i][j] where i=node(0..3), j=component(0=x,1=y,2=z)
     * @return QualityMetrics struct
     */
    static QualityMetrics compute_shell_quality(const Real coords[4][3]) {
        QualityMetrics q;

        // 4 edges
        Real e01 = edge_len(coords[0], coords[1]);
        Real e12 = edge_len(coords[1], coords[2]);
        Real e23 = edge_len(coords[2], coords[3]);
        Real e30 = edge_len(coords[3], coords[0]);

        Real min_e = std::min({e01, e12, e23, e30});
        Real max_e = std::max({e01, e12, e23, e30});
        q.aspect_ratio = (min_e > 1e-30) ? max_e / min_e : 1e30;

        // Warpage: angle between normals of the two triangles
        q.warpage = face_warpage(coords[0], coords[1], coords[2], coords[3]);

        // Interior angles at each corner
        Real angles[4];
        angles[0] = vertex_angle(coords[3], coords[0], coords[1]);
        angles[1] = vertex_angle(coords[0], coords[1], coords[2]);
        angles[2] = vertex_angle(coords[1], coords[2], coords[3]);
        angles[3] = vertex_angle(coords[2], coords[3], coords[0]);

        q.min_angle = *std::min_element(angles, angles + 4);
        q.max_angle = *std::max_element(angles, angles + 4);

        // Skewness: deviation of angles from ideal 90 degrees
        Real ideal = 90.0;
        Real worst = 0.0;
        for (Real a : angles) {
            worst = std::max(worst, std::abs(a - ideal));
        }
        q.skewness = worst / 90.0;
        q.skewness = std::min(1.0, q.skewness);

        // Check validity: area must be positive and angles must be reasonable
        // Compute area via cross product of diagonals
        Real d1x = coords[2][0] - coords[0][0];
        Real d1y = coords[2][1] - coords[0][1];
        Real d1z = coords[2][2] - coords[0][2];
        Real d2x = coords[3][0] - coords[1][0];
        Real d2y = coords[3][1] - coords[1][1];
        Real d2z = coords[3][2] - coords[1][2];
        Real nx = d1y * d2z - d1z * d2y;
        Real ny = d1z * d2x - d1x * d2z;
        Real nz = d1x * d2y - d1y * d2x;
        Real area = 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);

        q.jacobian_ratio = 1.0; // Shells: use warpage as primary quality measure
        q.is_valid = (area > 1e-30 && q.min_angle > 0.0 && q.max_angle < 180.0);

        return q;
    }

private:
    static Real edge_len(const Real a[3], const Real b[3]) {
        Real dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    static Real pt_dist(const Real a[3], const Real b[3]) {
        return edge_len(a, b);
    }

    /// Angle in degrees at vertex v, between edges v->prev and v->next
    static Real vertex_angle(const Real prev[3], const Real v[3], const Real next[3]) {
        Real ax = prev[0] - v[0], ay = prev[1] - v[1], az = prev[2] - v[2];
        Real bx = next[0] - v[0], by = next[1] - v[1], bz = next[2] - v[2];
        Real ma = std::sqrt(ax * ax + ay * ay + az * az);
        Real mb = std::sqrt(bx * bx + by * by + bz * bz);
        if (ma < 1e-30 || mb < 1e-30) return 0.0;
        Real dot = (ax * bx + ay * by + az * bz) / (ma * mb);
        dot = std::max(-1.0, std::min(1.0, dot));
        return std::acos(dot) * 180.0 / M_PI;
    }

    /// Warpage of a quad face: angle between normals of the two split triangles
    static Real face_warpage(const Real p0[3], const Real p1[3],
                             const Real p2[3], const Real p3[3]) {
        // Triangle 1: p0-p1-p2, Triangle 2: p0-p2-p3
        Real a1x = p1[0] - p0[0], a1y = p1[1] - p0[1], a1z = p1[2] - p0[2];
        Real b1x = p2[0] - p0[0], b1y = p2[1] - p0[1], b1z = p2[2] - p0[2];
        Real n1x = a1y * b1z - a1z * b1y;
        Real n1y = a1z * b1x - a1x * b1z;
        Real n1z = a1x * b1y - a1y * b1x;

        Real a2x = p2[0] - p0[0], a2y = p2[1] - p0[1], a2z = p2[2] - p0[2];
        Real b2x = p3[0] - p0[0], b2y = p3[1] - p0[1], b2z = p3[2] - p0[2];
        Real n2x = a2y * b2z - a2z * b2y;
        Real n2y = a2z * b2x - a2x * b2z;
        Real n2z = a2x * b2y - a2y * b2x;

        Real m1 = std::sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
        Real m2 = std::sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
        if (m1 < 1e-30 || m2 < 1e-30) return 0.0;

        Real dot = (n1x * n2x + n1y * n2y + n1z * n2z) / (m1 * m2);
        dot = std::max(-1.0, std::min(1.0, dot));
        return std::acos(dot) * 180.0 / M_PI;
    }
};

// ============================================================================
// 2. MeshRepair
// ============================================================================

/**
 * @brief Mesh repair utilities: duplicate node merging, free edge and T-junction detection
 *
 * Operates on simple node/element arrays for format independence.
 */
class MeshRepair {
public:
    /// Simple node representation for repair operations
    struct Node {
        int id;
        Real x, y, z;
    };

    /// Simple element representation for repair operations
    struct Element {
        int id;
        std::vector<int> node_ids;
    };

    /// Result of a merge operation: old_id -> kept_id
    struct MergeResult {
        std::vector<std::pair<int, int>> merge_map;
        int num_merged;
    };

    /// Edge represented as sorted pair of node IDs
    using Edge = std::pair<int, int>;

    /**
     * @brief Merge duplicate nodes within tolerance
     *
     * For each pair of nodes within `tolerance` distance, removes the later node
     * and updates the merge map. The caller should apply the merge map to element
     * connectivity afterward.
     *
     * Complexity: O(N^2) for small meshes; use spatial hashing for large meshes.
     *
     * @param nodes Node array (modified: duplicates removed)
     * @param tolerance Distance threshold for considering two nodes identical
     * @return MergeResult with mapping from removed IDs to surviving IDs
     */
    static MergeResult merge_duplicate_nodes(std::vector<Node>& nodes, Real tolerance = 1e-10) {
        MergeResult result;
        result.num_merged = 0;
        Real tol_sq = tolerance * tolerance;
        std::set<int> removed;

        for (size_t i = 0; i < nodes.size(); ++i) {
            if (removed.count(nodes[i].id)) continue;
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (removed.count(nodes[j].id)) continue;
                Real dx = nodes[j].x - nodes[i].x;
                Real dy = nodes[j].y - nodes[i].y;
                Real dz = nodes[j].z - nodes[i].z;
                if (dx * dx + dy * dy + dz * dz <= tol_sq) {
                    result.merge_map.push_back({nodes[j].id, nodes[i].id});
                    removed.insert(nodes[j].id);
                    result.num_merged++;
                }
            }
        }

        // Remove merged nodes
        if (result.num_merged > 0) {
            nodes.erase(
                std::remove_if(nodes.begin(), nodes.end(),
                    [&removed](const Node& n) { return removed.count(n.id) > 0; }),
                nodes.end());
        }

        return result;
    }

    /**
     * @brief Find free (boundary) edges in a mesh
     *
     * A free edge is shared by exactly one element. In a watertight surface mesh,
     * there are no free edges. Each edge is represented as a sorted pair (min_id, max_id).
     *
     * @param elements Element connectivity
     * @return Vector of free edges
     */
    static std::vector<Edge> find_free_edges(const std::vector<Element>& elements) {
        std::map<Edge, int> edge_count;
        for (const auto& elem : elements) {
            int n = static_cast<int>(elem.node_ids.size());
            for (int i = 0; i < n; ++i) {
                int a = elem.node_ids[i];
                int b = elem.node_ids[(i + 1) % n];
                Edge e = {std::min(a, b), std::max(a, b)};
                edge_count[e]++;
            }
        }

        std::vector<Edge> free;
        for (const auto& [edge, count] : edge_count) {
            if (count == 1) {
                free.push_back(edge);
            }
        }
        return free;
    }

    /**
     * @brief Find T-junctions in a mesh
     *
     * A T-junction is an edge shared by more than 2 elements, indicating a
     * non-manifold topology where one element terminates mid-face of another.
     *
     * @param elements Element connectivity
     * @return Vector of T-junction edges
     */
    static std::vector<Edge> find_t_junctions(const std::vector<Element>& elements) {
        std::map<Edge, int> edge_count;
        for (const auto& elem : elements) {
            int n = static_cast<int>(elem.node_ids.size());
            for (int i = 0; i < n; ++i) {
                int a = elem.node_ids[i];
                int b = elem.node_ids[(i + 1) % n];
                Edge e = {std::min(a, b), std::max(a, b)};
                edge_count[e]++;
            }
        }

        std::vector<Edge> tjunc;
        for (const auto& [edge, count] : edge_count) {
            if (count > 2) {
                tjunc.push_back(edge);
            }
        }
        return tjunc;
    }

    /**
     * @brief Apply a merge map to element connectivity
     *
     * Replaces all occurrences of old node IDs with their surviving counterparts.
     *
     * @param elements Element connectivity (modified in place)
     * @param merge_map Vector of (old_id, kept_id) pairs
     */
    static void apply_merge_to_elements(std::vector<Element>& elements,
                                         const std::vector<std::pair<int, int>>& merge_map) {
        std::map<int, int> lookup;
        for (const auto& [old_id, kept_id] : merge_map) {
            lookup[old_id] = kept_id;
        }
        for (auto& elem : elements) {
            for (auto& nid : elem.node_ids) {
                auto it = lookup.find(nid);
                if (it != lookup.end()) {
                    nid = it->second;
                }
            }
        }
    }
};

// ============================================================================
// 3. AutoContactSurface
// ============================================================================

/**
 * @brief Exterior face data
 */
struct ExteriorFace {
    int node_ids[4];    ///< Face node IDs (node_ids[3]=0 for triangles)
    int num_nodes;      ///< 3 or 4
    int element_id;     ///< Parent element
    int part_id;        ///< Part this face belongs to
    Real normal[3];     ///< Outward unit normal
};

/**
 * @brief Automatic contact surface detection from solid/shell boundaries
 *
 * Finds exterior faces of a solid mesh (faces shared by only one element),
 * groups them by part, and creates contact pairs between adjacent parts.
 */
class AutoContactSurface {
public:
    /// A contact pair between two parts
    struct ContactPair {
        int part1;
        int part2;
    };

    /**
     * @brief Extract exterior faces from solid elements
     *
     * For hex elements: a face is exterior if it appears in only one element.
     * For shell elements: both faces are exterior by definition.
     *
     * @param elements Vector of elements. Each element has id, node_ids (8 for hex),
     *        and a part_id field (stored in element struct or separate map).
     * @param node_ids Element connectivity as flat array: node_ids[elem_idx * max_nodes + local]
     * @param num_nodes_per_elem Number of nodes per element
     * @param num_elements Total number of elements
     * @param part_ids Part ID for each element (size = num_elements)
     * @return Vector of exterior faces
     */
    static std::vector<ExteriorFace> extract_exterior_faces(
        const int* node_ids, int num_nodes_per_elem, int num_elements,
        const int* part_ids,
        const Real* node_coords_x, const Real* node_coords_y, const Real* node_coords_z,
        const std::map<int, int>& node_id_to_idx)
    {
        // Hex face local node indices
        static const int hex_faces[6][4] = {
            {0,1,2,3}, {4,7,6,5}, {0,4,5,1},
            {2,6,7,3}, {0,3,7,4}, {1,5,6,2}
        };

        // Build face -> count map using sorted 4-node key
        using FaceKey = std::array<int, 4>;
        std::map<FaceKey, int> face_count;
        std::map<FaceKey, ExteriorFace> face_data;

        for (int e = 0; e < num_elements; ++e) {
            const int* en = node_ids + e * num_nodes_per_elem;
            int pid = part_ids[e];

            if (num_nodes_per_elem >= 8) {
                // Hex element: 6 faces
                for (int f = 0; f < 6; ++f) {
                    ExteriorFace ef;
                    ef.num_nodes = 4;
                    ef.element_id = e;
                    ef.part_id = pid;
                    for (int k = 0; k < 4; ++k) {
                        ef.node_ids[k] = en[hex_faces[f][k]];
                    }

                    compute_normal(ef, node_coords_x, node_coords_y, node_coords_z,
                                   node_id_to_idx);

                    FaceKey key = {ef.node_ids[0], ef.node_ids[1],
                                   ef.node_ids[2], ef.node_ids[3]};
                    std::sort(key.begin(), key.end());
                    face_count[key]++;
                    face_data[key] = ef;
                }
            } else if (num_nodes_per_elem >= 4) {
                // Shell element: one face (the element itself)
                ExteriorFace ef;
                ef.num_nodes = std::min(num_nodes_per_elem, 4);
                ef.element_id = e;
                ef.part_id = pid;
                for (int k = 0; k < ef.num_nodes; ++k) {
                    ef.node_ids[k] = en[k];
                }
                if (ef.num_nodes == 3) ef.node_ids[3] = 0;

                compute_normal(ef, node_coords_x, node_coords_y, node_coords_z,
                               node_id_to_idx);

                FaceKey key = {ef.node_ids[0], ef.node_ids[1],
                               ef.node_ids[2], ef.node_ids[3]};
                std::sort(key.begin(), key.end());
                face_count[key]++;
                face_data[key] = ef;
            }
        }

        // Collect exterior faces (count == 1)
        std::vector<ExteriorFace> exterior;
        for (const auto& [key, count] : face_count) {
            if (count == 1) {
                exterior.push_back(face_data[key]);
            }
        }

        return exterior;
    }

    /**
     * @brief Group exterior faces by part ID
     * @param faces Vector of exterior faces
     * @return Map from part_id to vector of faces belonging to that part
     */
    static std::map<int, std::vector<ExteriorFace>> group_by_part(
        const std::vector<ExteriorFace>& faces)
    {
        std::map<int, std::vector<ExteriorFace>> grouped;
        for (const auto& f : faces) {
            grouped[f.part_id].push_back(f);
        }
        return grouped;
    }

    /**
     * @brief Create contact pairs between all combinations of exterior surface parts
     * @param part1 First part ID
     * @param part2 Second part ID
     * @return A ContactPair struct
     */
    static ContactPair create_contact_pair(int part1, int part2) {
        return {part1, part2};
    }

    /**
     * @brief Generate all contact pairs from grouped faces
     * @param faces_by_part Grouped faces from group_by_part()
     * @return Vector of contact pairs (one per unique part combination)
     */
    static std::vector<ContactPair> generate_all_pairs(
        const std::map<int, std::vector<ExteriorFace>>& faces_by_part)
    {
        std::vector<int> parts;
        for (const auto& [pid, _] : faces_by_part) {
            parts.push_back(pid);
        }

        std::vector<ContactPair> pairs;
        for (size_t i = 0; i < parts.size(); ++i) {
            for (size_t j = i + 1; j < parts.size(); ++j) {
                pairs.push_back({parts[i], parts[j]});
            }
        }
        return pairs;
    }

private:
    /// Compute outward normal for a face
    static void compute_normal(ExteriorFace& face,
                                const Real* cx, const Real* cy, const Real* cz,
                                const std::map<int, int>& nmap) {
        face.normal[0] = face.normal[1] = face.normal[2] = 0.0;
        if (face.num_nodes < 3) return;

        auto get = [&](int nid, Real& px, Real& py, Real& pz) -> bool {
            auto it = nmap.find(nid);
            if (it == nmap.end()) return false;
            int idx = it->second;
            px = cx[idx]; py = cy[idx]; pz = cz[idx];
            return true;
        };

        Real x0, y0, z0, x1, y1, z1, x2, y2, z2;
        if (!get(face.node_ids[0], x0, y0, z0)) return;
        if (!get(face.node_ids[1], x1, y1, z1)) return;
        if (!get(face.node_ids[2], x2, y2, z2)) return;

        Real ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
        Real bx = x2 - x0, by = y2 - y0, bz = z2 - z0;

        Real nx = ay * bz - az * by;
        Real ny = az * bx - ax * bz;
        Real nz = ax * by - ay * bx;

        Real mag = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (mag > 1e-30) {
            face.normal[0] = nx / mag;
            face.normal[1] = ny / mag;
            face.normal[2] = nz / mag;
        }
    }
};

// ============================================================================
// 4. PartMaterialAssigner
// ============================================================================

/**
 * @brief Part and material assignment helpers
 *
 * Provides methods to assign materials to parts and to assign elements to parts
 * by geometric region (box, sphere, or cylinder).
 */
class PartMaterialAssigner {
public:
    /// Assignment record
    struct Assignment {
        int element_id;
        int part_id;
    };

    /**
     * @brief Assign a material to a part (returns the mapping)
     * @param part_id Target part
     * @param mat_id Material to assign
     * @return Pair of (part_id, mat_id)
     */
    static std::pair<int, int> assign_material_to_part(int part_id, int mat_id) {
        return {part_id, mat_id};
    }

    /**
     * @brief Assign elements to a part if their centroid lies inside a box
     * @param elem_ids Element IDs
     * @param elem_node_ids Connectivity: elem_node_ids[i] = vector of node IDs for element i
     * @param node_x, node_y, node_z Node coordinate arrays
     * @param node_map node_id -> array index
     * @param box_min Minimum corner {xmin, ymin, zmin}
     * @param box_max Maximum corner {xmax, ymax, zmax}
     * @param part_id Part to assign
     * @return Vector of assignments
     */
    static std::vector<Assignment> assign_by_box(
        const std::vector<int>& elem_ids,
        const std::vector<std::vector<int>>& elem_node_ids,
        const Real* node_x, const Real* node_y, const Real* node_z,
        const std::map<int, int>& node_map,
        const Real box_min[3], const Real box_max[3], int part_id)
    {
        std::vector<Assignment> result;
        for (size_t e = 0; e < elem_ids.size(); ++e) {
            Real cx = 0.0, cy = 0.0, cz = 0.0;
            int count = 0;
            for (int nid : elem_node_ids[e]) {
                auto it = node_map.find(nid);
                if (it != node_map.end()) {
                    int idx = it->second;
                    cx += node_x[idx];
                    cy += node_y[idx];
                    cz += node_z[idx];
                    count++;
                }
            }
            if (count == 0) continue;
            cx /= count; cy /= count; cz /= count;

            if (cx >= box_min[0] && cx <= box_max[0] &&
                cy >= box_min[1] && cy <= box_max[1] &&
                cz >= box_min[2] && cz <= box_max[2]) {
                result.push_back({elem_ids[e], part_id});
            }
        }
        return result;
    }

    /**
     * @brief Assign elements to a part if their centroid lies inside a sphere
     * @param center Sphere center {x, y, z}
     * @param radius Sphere radius
     */
    static std::vector<Assignment> assign_by_sphere(
        const std::vector<int>& elem_ids,
        const std::vector<std::vector<int>>& elem_node_ids,
        const Real* node_x, const Real* node_y, const Real* node_z,
        const std::map<int, int>& node_map,
        const Real center[3], Real radius, int part_id)
    {
        std::vector<Assignment> result;
        Real r_sq = radius * radius;
        for (size_t e = 0; e < elem_ids.size(); ++e) {
            Real cx = 0.0, cy = 0.0, cz = 0.0;
            int count = 0;
            for (int nid : elem_node_ids[e]) {
                auto it = node_map.find(nid);
                if (it != node_map.end()) {
                    int idx = it->second;
                    cx += node_x[idx];
                    cy += node_y[idx];
                    cz += node_z[idx];
                    count++;
                }
            }
            if (count == 0) continue;
            cx /= count; cy /= count; cz /= count;

            Real dx = cx - center[0], dy = cy - center[1], dz = cz - center[2];
            if (dx * dx + dy * dy + dz * dz <= r_sq) {
                result.push_back({elem_ids[e], part_id});
            }
        }
        return result;
    }

    /**
     * @brief Assign elements to a part if their centroid lies inside a cylinder
     * @param origin Cylinder axis origin point
     * @param axis Cylinder axis direction (unit vector)
     * @param radius Cylinder radius
     * @param half_length Half-length of the cylinder along the axis
     */
    static std::vector<Assignment> assign_by_cylinder(
        const std::vector<int>& elem_ids,
        const std::vector<std::vector<int>>& elem_node_ids,
        const Real* node_x, const Real* node_y, const Real* node_z,
        const std::map<int, int>& node_map,
        const Real origin[3], const Real axis[3], Real radius, Real half_length,
        int part_id)
    {
        std::vector<Assignment> result;
        Real r_sq = radius * radius;

        // Normalize axis
        Real amag = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        Real ax = axis[0] / amag, ay = axis[1] / amag, az = axis[2] / amag;

        for (size_t e = 0; e < elem_ids.size(); ++e) {
            Real cx = 0.0, cy = 0.0, cz = 0.0;
            int count = 0;
            for (int nid : elem_node_ids[e]) {
                auto it = node_map.find(nid);
                if (it != node_map.end()) {
                    int idx = it->second;
                    cx += node_x[idx];
                    cy += node_y[idx];
                    cz += node_z[idx];
                    count++;
                }
            }
            if (count == 0) continue;
            cx /= count; cy /= count; cz /= count;

            // Vector from origin to centroid
            Real vx = cx - origin[0], vy = cy - origin[1], vz = cz - origin[2];
            // Project onto axis
            Real proj = vx * ax + vy * ay + vz * az;
            if (std::abs(proj) > half_length) continue;
            // Perpendicular distance squared
            Real perp_x = vx - proj * ax;
            Real perp_y = vy - proj * ay;
            Real perp_z = vz - proj * az;
            Real perp_sq = perp_x * perp_x + perp_y * perp_y + perp_z * perp_z;
            if (perp_sq <= r_sq) {
                result.push_back({elem_ids[e], part_id});
            }
        }
        return result;
    }
};

// ============================================================================
// 5. CoordinateTransform
// ============================================================================

/**
 * @brief Coordinate system transformations: rectangular, cylindrical, spherical
 *
 * Transforms node positions, vectors, and symmetric tensors between coordinate systems.
 * Cylindrical: (r, theta, z) where theta is measured from the x-axis in the xy-plane.
 * Spherical: (r, theta, phi) where theta is polar (from z-axis) and phi is azimuthal.
 */
class CoordinateTransform {
public:
    /**
     * @brief Transform nodes from Cartesian to cylindrical coordinates
     *
     * @param nodes_x, nodes_y, nodes_z Input Cartesian (modified to output cylindrical: r, theta, z)
     * @param num_nodes Number of nodes
     * @param origin Axis origin [3]
     * @param axis Axis direction [3] (the z-axis of the cylindrical system)
     */
    static void to_cylindrical(Real* nodes_x, Real* nodes_y, Real* nodes_z,
                                int num_nodes,
                                const Real origin[3], const Real axis[3]) {
        // Build orthonormal frame: e3 = axis, e1 = arbitrary perp, e2 = e3 x e1
        Real e3[3];
        Real amag = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        e3[0] = axis[0] / amag; e3[1] = axis[1] / amag; e3[2] = axis[2] / amag;

        Real e1[3], e2[3];
        build_perp_frame(e3, e1, e2);

        for (int i = 0; i < num_nodes; ++i) {
            Real dx = nodes_x[i] - origin[0];
            Real dy = nodes_y[i] - origin[1];
            Real dz = nodes_z[i] - origin[2];

            // Project onto local frame
            Real p1 = dx * e1[0] + dy * e1[1] + dz * e1[2];
            Real p2 = dx * e2[0] + dy * e2[1] + dz * e2[2];
            Real p3 = dx * e3[0] + dy * e3[1] + dz * e3[2];

            nodes_x[i] = std::sqrt(p1 * p1 + p2 * p2);   // r
            nodes_y[i] = std::atan2(p2, p1);               // theta
            nodes_z[i] = p3;                                // z (along axis)
        }
    }

    /**
     * @brief Transform nodes from Cartesian to spherical coordinates
     *
     * @param nodes_x, nodes_y, nodes_z Input Cartesian (modified to r, theta, phi)
     * @param num_nodes Number of nodes
     * @param origin Sphere center [3]
     */
    static void to_spherical(Real* nodes_x, Real* nodes_y, Real* nodes_z,
                              int num_nodes, const Real origin[3]) {
        for (int i = 0; i < num_nodes; ++i) {
            Real dx = nodes_x[i] - origin[0];
            Real dy = nodes_y[i] - origin[1];
            Real dz = nodes_z[i] - origin[2];

            Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            Real theta = (r > 1e-30) ? std::acos(std::max(-1.0, std::min(1.0, dz / r))) : 0.0;
            Real phi = std::atan2(dy, dx);

            nodes_x[i] = r;
            nodes_y[i] = theta;
            nodes_z[i] = phi;
        }
    }

    /**
     * @brief Transform nodes from cylindrical back to Cartesian
     *
     * @param nodes_x, nodes_y, nodes_z Input cylindrical (r, theta, z) -> output Cartesian
     * @param num_nodes Number of nodes
     * @param origin Axis origin [3]
     * @param axis Axis direction [3]
     */
    static void from_cylindrical(Real* nodes_x, Real* nodes_y, Real* nodes_z,
                                  int num_nodes,
                                  const Real origin[3], const Real axis[3]) {
        Real e3[3];
        Real amag = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        e3[0] = axis[0] / amag; e3[1] = axis[1] / amag; e3[2] = axis[2] / amag;

        Real e1[3], e2[3];
        build_perp_frame(e3, e1, e2);

        for (int i = 0; i < num_nodes; ++i) {
            Real r     = nodes_x[i];
            Real theta = nodes_y[i];
            Real z     = nodes_z[i];

            Real p1 = r * std::cos(theta);
            Real p2 = r * std::sin(theta);
            Real p3 = z;

            nodes_x[i] = origin[0] + p1 * e1[0] + p2 * e2[0] + p3 * e3[0];
            nodes_y[i] = origin[1] + p1 * e1[1] + p2 * e2[1] + p3 * e3[1];
            nodes_z[i] = origin[2] + p1 * e1[2] + p2 * e2[2] + p3 * e3[2];
        }
    }

    /**
     * @brief Transform nodes from spherical back to Cartesian
     */
    static void from_spherical(Real* nodes_x, Real* nodes_y, Real* nodes_z,
                                int num_nodes, const Real origin[3]) {
        for (int i = 0; i < num_nodes; ++i) {
            Real r     = nodes_x[i];
            Real theta = nodes_y[i];
            Real phi   = nodes_z[i];

            nodes_x[i] = origin[0] + r * std::sin(theta) * std::cos(phi);
            nodes_y[i] = origin[1] + r * std::sin(theta) * std::sin(phi);
            nodes_z[i] = origin[2] + r * std::cos(theta);
        }
    }

    /**
     * @brief Transform a symmetric 3x3 tensor (Voigt: xx, yy, zz, xy, yz, xz)
     *        by rotation matrix Q: T' = Q * T * Q^T
     *
     * @param tensor Input Voigt tensor [6]
     * @param rotation_matrix 3x3 rotation matrix (row-major, 9 values)
     * @param result Output Voigt tensor [6]
     */
    static void transform_tensor(const Real tensor[6], const Real rotation_matrix[9],
                                  Real result[6]) {
        // Expand Voigt to full 3x3
        Real T[3][3];
        T[0][0] = tensor[0]; T[1][1] = tensor[1]; T[2][2] = tensor[2];
        T[0][1] = tensor[3]; T[1][0] = tensor[3];
        T[1][2] = tensor[4]; T[2][1] = tensor[4];
        T[0][2] = tensor[5]; T[2][0] = tensor[5];

        // Q stored row-major: Q[i][j] = rotation_matrix[i*3+j]
        // Compute Tp = Q * T * Q^T
        Real Tp[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Tp[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        Tp[i][j] += rotation_matrix[i * 3 + k] * T[k][l]
                                  * rotation_matrix[j * 3 + l];
                    }
                }
            }
        }

        // Pack back to Voigt
        result[0] = Tp[0][0]; result[1] = Tp[1][1]; result[2] = Tp[2][2];
        result[3] = Tp[0][1]; result[4] = Tp[1][2]; result[5] = Tp[0][2];
    }

    /**
     * @brief Transform a vector by rotation matrix: v' = Q * v
     * @param vec Input vector [3]
     * @param rotation_matrix 3x3 rotation matrix (row-major, 9 values)
     * @param result Output vector [3]
     */
    static void transform_vector(const Real vec[3], const Real rotation_matrix[9],
                                  Real result[3]) {
        for (int i = 0; i < 3; ++i) {
            result[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                result[i] += rotation_matrix[i * 3 + j] * vec[j];
            }
        }
    }

    /**
     * @brief Build rotation matrix for cylindrical-to-Cartesian at a given angle theta
     *
     * The rotation matrix maps (e_r, e_theta, e_z) to (e_x, e_y, e_z):
     *   Q = [[cos(theta), -sin(theta), 0],
     *        [sin(theta),  cos(theta), 0],
     *        [0,           0,          1]]
     *
     * @param theta Azimuthal angle in radians
     * @param Q Output 3x3 rotation matrix (row-major, 9 values)
     */
    static void cylindrical_rotation_matrix(Real theta, Real Q[9]) {
        Real ct = std::cos(theta), st = std::sin(theta);
        Q[0] = ct;  Q[1] = -st; Q[2] = 0.0;
        Q[3] = st;  Q[4] = ct;  Q[5] = 0.0;
        Q[6] = 0.0; Q[7] = 0.0; Q[8] = 1.0;
    }

    /**
     * @brief Build rotation matrix for spherical-to-Cartesian at given (theta, phi)
     *
     * Maps (e_r, e_theta, e_phi) to (e_x, e_y, e_z).
     *
     * @param theta Polar angle (from z-axis) in radians
     * @param phi Azimuthal angle (from x-axis in xy-plane) in radians
     * @param Q Output 3x3 rotation matrix (row-major, 9 values)
     */
    static void spherical_rotation_matrix(Real theta, Real phi, Real Q[9]) {
        Real st = std::sin(theta), ct = std::cos(theta);
        Real sp = std::sin(phi),   cp = std::cos(phi);

        // e_r     = (st*cp, st*sp, ct)
        // e_theta = (ct*cp, ct*sp, -st)
        // e_phi   = (-sp,   cp,    0)
        Q[0] = st * cp; Q[1] = ct * cp; Q[2] = -sp;
        Q[3] = st * sp; Q[4] = ct * sp; Q[5] = cp;
        Q[6] = ct;      Q[7] = -st;     Q[8] = 0.0;
    }

private:
    /// Build an orthonormal frame given e3: find e1 perp to e3, then e2 = e3 x e1
    static void build_perp_frame(const Real e3[3], Real e1[3], Real e2[3]) {
        // Choose the axis least parallel to e3 as seed
        Real absx = std::abs(e3[0]), absy = std::abs(e3[1]), absz = std::abs(e3[2]);
        Real seed[3];
        if (absx <= absy && absx <= absz) {
            seed[0] = 1.0; seed[1] = 0.0; seed[2] = 0.0;
        } else if (absy <= absz) {
            seed[0] = 0.0; seed[1] = 1.0; seed[2] = 0.0;
        } else {
            seed[0] = 0.0; seed[1] = 0.0; seed[2] = 1.0;
        }

        // e1 = seed - (seed . e3) * e3, then normalize
        Real dot = seed[0] * e3[0] + seed[1] * e3[1] + seed[2] * e3[2];
        e1[0] = seed[0] - dot * e3[0];
        e1[1] = seed[1] - dot * e3[1];
        e1[2] = seed[2] - dot * e3[2];
        Real m = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        e1[0] /= m; e1[1] /= m; e1[2] /= m;

        // e2 = e3 x e1
        e2[0] = e3[1] * e1[2] - e3[2] * e1[1];
        e2[1] = e3[2] * e1[0] - e3[0] * e1[2];
        e2[2] = e3[0] * e1[1] - e3[1] * e1[0];
    }
};

} // namespace io
} // namespace nxs
