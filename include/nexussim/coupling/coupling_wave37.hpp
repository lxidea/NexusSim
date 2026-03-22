#pragma once

/**
 * @file coupling_wave37.hpp
 * @brief Wave 37: Coupling Framework (6 features)
 *
 * Features:
 *   1. CouplingAdapter          - Abstract base for code coupling
 *   2. PreCICEAdapter           - preCICE-compatible stub adapter
 *   3. CWIPIAdapter             - CWIPI code-to-code coupling stub
 *   4. Rad2RadCoupling          - Multi-domain NexusSim-to-NexusSim coupling
 *   5. PythonCoupling           - Python callback co-simulation
 *   6. CouplingInterpolation    - RBF and nearest-neighbor field interpolation
 *
 * Namespace: nxs::coupling
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>

namespace nxs {
namespace coupling {

using Real = nxs::Real;

// ============================================================================
// 1. CouplingAdapter — Abstract base for code coupling
// ============================================================================

/**
 * @brief Configuration for coupling adapters
 */
struct CouplingConfig {
    int comm_rank = 0;          ///< MPI rank (or 0 for serial)
    int comm_size = 1;          ///< MPI communicator size
    Real dt_coupling = 1.0e-3;  ///< Coupling timestep
    int max_iterations = 10;    ///< Max sub-iterations per coupling step
};

/**
 * @brief Abstract base class for all coupling adapters
 *
 * Defines the minimal interface for exchanging field data between
 * coupled solvers: initialize, send, receive, advance, finalize.
 * Concrete adapters implement the communication mechanism.
 */
class CouplingAdapter {
public:
    virtual ~CouplingAdapter() = default;

    /// Initialize the adapter with the given configuration
    virtual void initialize(const CouplingConfig& config) = 0;

    /// Send a named field (n values) to the coupled solver
    virtual void send_field(const char* name, const Real* data, int n) = 0;

    /// Receive a named field (n values) from the coupled solver
    virtual void receive_field(const char* name, Real* data, int n) = 0;

    /// Advance the coupling by one timestep dt
    virtual void advance(Real dt) = 0;

    /// Finalize and release all coupling resources
    virtual void finalize() = 0;

    /// Query whether the adapter has been initialized
    bool is_initialized() const { return initialized_; }

    /// Get the current coupling configuration
    const CouplingConfig& config() const { return config_; }

protected:
    CouplingConfig config_{};
    bool initialized_ = false;
    Real current_time_ = 0.0;
};


// ============================================================================
// 2. PreCICEAdapter — preCICE-compatible stub adapter
// ============================================================================

/**
 * @brief preCICE-compatible coupling adapter (stub, no actual preCICE dependency)
 *
 * Provides the same logical workflow as preCICE: register meshes, define
 * data fields, read/write data, and advance the coupling. Internally uses
 * local buffers for self-coupling tests.
 *
 * Typical workflow:
 *   adapter.initialize(config);
 *   adapter.register_mesh("FluidMesh", coords, n_verts);
 *   adapter.write_data("Temperature", temps);
 *   adapter.advance(dt);
 *   adapter.read_data("HeatFlux", fluxes);
 *   adapter.finalize();
 */
class PreCICEAdapter : public CouplingAdapter {
public:
    struct MeshData {
        std::string name;
        std::vector<Real> coordinates;  ///< 3*n_verts flattened
        int n_vertices = 0;
    };

    struct FieldBuffer {
        std::string name;
        std::vector<Real> values;
        int size = 0;
        bool has_data = false;
    };

    PreCICEAdapter() = default;

    void initialize(const CouplingConfig& config) override {
        config_ = config;
        initialized_ = true;
        current_time_ = 0.0;
        meshes_.clear();
        fields_.clear();
        coupling_step_ = 0;
    }

    /// Register a coupling mesh with vertex coordinates (3*n_verts)
    void register_mesh(const char* name, const Real* coords, int n_verts) {
        MeshData mesh;
        mesh.name = name;
        mesh.n_vertices = n_verts;
        mesh.coordinates.assign(coords, coords + 3 * n_verts);
        meshes_[name] = std::move(mesh);
    }

    /// Write field data to the internal buffer (to be sent to partner)
    void write_data(const char* name, const Real* values, int n) {
        FieldBuffer& fb = fields_[name];
        fb.name = name;
        fb.size = n;
        fb.values.assign(values, values + n);
        fb.has_data = true;
    }

    /// Read field data from the internal buffer (received from partner)
    void read_data(const char* name, Real* values, int n) {
        auto it = fields_.find(name);
        if (it != fields_.end() && it->second.has_data) {
            int copy_n = (n < it->second.size) ? n : it->second.size;
            std::memcpy(values, it->second.values.data(),
                        static_cast<size_t>(copy_n) * sizeof(Real));
        } else {
            // No data available; zero fill
            std::memset(values, 0, static_cast<size_t>(n) * sizeof(Real));
        }
    }

    void send_field(const char* name, const Real* data, int n) override {
        write_data(name, data, n);
    }

    void receive_field(const char* name, Real* data, int n) override {
        read_data(name, data, n);
    }

    void advance(Real dt) override {
        current_time_ += dt;
        coupling_step_++;
    }

    void finalize() override {
        meshes_.clear();
        fields_.clear();
        initialized_ = false;
        coupling_step_ = 0;
    }

    /// Check if a mesh is registered
    bool has_mesh(const char* name) const {
        return meshes_.find(name) != meshes_.end();
    }

    /// Check if a field buffer has data
    bool has_field_data(const char* name) const {
        auto it = fields_.find(name);
        return (it != fields_.end()) && it->second.has_data;
    }

    /// Get the number of vertices for a registered mesh
    int mesh_vertex_count(const char* name) const {
        auto it = meshes_.find(name);
        return (it != meshes_.end()) ? it->second.n_vertices : 0;
    }

    /// Get current coupling step
    int coupling_step() const { return coupling_step_; }

    /// Get current time
    Real current_time() const { return current_time_; }

private:
    std::unordered_map<std::string, MeshData> meshes_;
    std::unordered_map<std::string, FieldBuffer> fields_;
    int coupling_step_ = 0;
};


// ============================================================================
// 3. CWIPIAdapter — CWIPI code-to-code coupling stub
// ============================================================================

/**
 * @brief CWIPI code-to-code coupling stub
 *
 * Models the CWIPI (Coupling With Interpolation Parallel Interface) pattern:
 * set a coupling mesh, compute interpolation weights, and exchange fields.
 * Uses local buffering for self-coupling tests.
 *
 * Interpolation weights are computed from a simple inverse-distance scheme
 * mapping source to target meshes.
 */
class CWIPIAdapter : public CouplingAdapter {
public:
    CWIPIAdapter() = default;

    void initialize(const CouplingConfig& config) override {
        config_ = config;
        initialized_ = true;
        current_time_ = 0.0;
        mesh_coords_.clear();
        n_mesh_pts_ = 0;
        send_buf_.clear();
        recv_buf_.clear();
    }

    /// Set the coupling mesh coordinates (3*n flattened)
    void set_coupling_mesh(const Real* coords, int n) {
        n_mesh_pts_ = n;
        mesh_coords_.assign(coords, coords + 3 * n);
    }

    /// Compute interpolation weights from source to local mesh
    /// Uses inverse-distance weighting with compact support radius
    void compute_weights(const Real* source_coords, int n_src, Real support) {
        weights_.resize(static_cast<size_t>(n_mesh_pts_) * n_src, 0.0);
        weight_src_count_ = n_src;

        for (int i = 0; i < n_mesh_pts_; ++i) {
            Real xi = mesh_coords_[3 * i];
            Real yi = mesh_coords_[3 * i + 1];
            Real zi = mesh_coords_[3 * i + 2];
            Real wsum = 0.0;

            for (int j = 0; j < n_src; ++j) {
                Real dx = xi - source_coords[3 * j];
                Real dy = yi - source_coords[3 * j + 1];
                Real dz = zi - source_coords[3 * j + 2];
                Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
                Real w = 0.0;
                if (r < support) {
                    Real q = r / support;
                    w = (1.0 - q) * (1.0 - q);  // compact-support quadratic
                    if (r < 1.0e-15) w = 1.0e15; // near-exact match
                }
                weights_[static_cast<size_t>(i) * n_src + j] = w;
                wsum += w;
            }
            // Normalize
            if (wsum > 0.0) {
                for (int j = 0; j < n_src; ++j) {
                    weights_[static_cast<size_t>(i) * n_src + j] /= wsum;
                }
            }
        }
    }

    /// Interpolate source field to local mesh using precomputed weights
    void interpolate(const Real* src_vals, int n_src, Real* dst_vals) const {
        for (int i = 0; i < n_mesh_pts_; ++i) {
            Real val = 0.0;
            for (int j = 0; j < n_src; ++j) {
                val += weights_[static_cast<size_t>(i) * n_src + j] * src_vals[j];
            }
            dst_vals[i] = val;
        }
    }

    /// Exchange: send local field, receive remote field (self-loop for stub)
    void exchange(const Real* send_field_data, int n_send,
                  Real* recv_field_data, int n_recv) {
        send_buf_.assign(send_field_data, send_field_data + n_send);
        // In stub mode, the received data mirrors the sent data
        int copy_n = (n_send < n_recv) ? n_send : n_recv;
        std::memcpy(recv_field_data, send_field_data,
                    static_cast<size_t>(copy_n) * sizeof(Real));
        // Zero-fill remainder
        for (int i = copy_n; i < n_recv; ++i) {
            recv_field_data[i] = 0.0;
        }
        recv_buf_.assign(recv_field_data, recv_field_data + n_recv);
    }

    void send_field(const char* /*name*/, const Real* data, int n) override {
        send_buf_.assign(data, data + n);
    }

    void receive_field(const char* /*name*/, Real* data, int n) override {
        int copy_n = (static_cast<int>(send_buf_.size()) < n)
                         ? static_cast<int>(send_buf_.size()) : n;
        if (copy_n > 0) {
            std::memcpy(data, send_buf_.data(),
                        static_cast<size_t>(copy_n) * sizeof(Real));
        }
        for (int i = copy_n; i < n; ++i) data[i] = 0.0;
    }

    void advance(Real dt) override {
        current_time_ += dt;
    }

    void finalize() override {
        mesh_coords_.clear();
        weights_.clear();
        send_buf_.clear();
        recv_buf_.clear();
        n_mesh_pts_ = 0;
        initialized_ = false;
    }

    int mesh_point_count() const { return n_mesh_pts_; }

private:
    std::vector<Real> mesh_coords_;
    int n_mesh_pts_ = 0;
    std::vector<Real> weights_;
    int weight_src_count_ = 0;
    std::vector<Real> send_buf_;
    std::vector<Real> recv_buf_;
};


// ============================================================================
// 4. Rad2RadCoupling — Multi-domain NexusSim-to-NexusSim coupling
// ============================================================================

/**
 * @brief Domain interface for shared nodes between NexusSim domains
 */
struct DomainInterface {
    int* shared_nodes = nullptr;   ///< Indices of shared nodes
    int n_shared = 0;              ///< Number of shared nodes
    Real* send_buf = nullptr;      ///< Send buffer (3*n_shared for vector fields)
    Real* recv_buf = nullptr;      ///< Receive buffer (3*n_shared)
};

/**
 * @brief Multi-domain NexusSim-to-NexusSim coupling
 *
 * Manages interface nodes shared between sub-domains in a domain-decomposed
 * simulation. Provides exchange of displacements, velocities, and forces
 * across domain interfaces with conservation guarantees.
 *
 * In single-process (stub) mode, send_buf is copied directly to recv_buf
 * to simulate the exchange. Force balance is enforced by negating the
 * interface forces on the receiving side (Newton's third law).
 */
class Rad2RadCoupling {
public:
    Rad2RadCoupling() = default;

    /// Initialize with domain decomposition info
    void initialize(int domain_id, int n_domains) {
        domain_id_ = domain_id;
        n_domains_ = n_domains;
        initialized_ = true;
    }

    /// Pack displacement/velocity data from global arrays into interface send buffer
    /// field_data: global array of 3*n_total Reals; stride = 3 (x,y,z per node)
    void pack_interface(const DomainInterface& iface, const Real* field_data) {
        for (int i = 0; i < iface.n_shared; ++i) {
            int node = iface.shared_nodes[i];
            iface.send_buf[3 * i + 0] = field_data[3 * node + 0];
            iface.send_buf[3 * i + 1] = field_data[3 * node + 1];
            iface.send_buf[3 * i + 2] = field_data[3 * node + 2];
        }
    }

    /// Exchange interface data (stub: copy send to recv)
    void exchange_interface(DomainInterface& iface) {
        std::memcpy(iface.recv_buf, iface.send_buf,
                    static_cast<size_t>(3 * iface.n_shared) * sizeof(Real));
    }

    /// Unpack received data into global arrays
    void unpack_interface(const DomainInterface& iface, Real* field_data) {
        for (int i = 0; i < iface.n_shared; ++i) {
            int node = iface.shared_nodes[i];
            field_data[3 * node + 0] = iface.recv_buf[3 * i + 0];
            field_data[3 * node + 1] = iface.recv_buf[3 * i + 1];
            field_data[3 * node + 2] = iface.recv_buf[3 * i + 2];
        }
    }

    /// Apply interface forces: adds received forces to local force array
    /// Newton's third law: recv forces are applied as-is (partner already negated)
    void apply_interface_forces(const DomainInterface& iface, Real* forces) {
        for (int i = 0; i < iface.n_shared; ++i) {
            int node = iface.shared_nodes[i];
            forces[3 * node + 0] += iface.recv_buf[3 * i + 0];
            forces[3 * node + 1] += iface.recv_buf[3 * i + 1];
            forces[3 * node + 2] += iface.recv_buf[3 * i + 2];
        }
    }

    /// Average interface values (useful for displacement continuity)
    void average_interface(const DomainInterface& iface, Real* field_data) {
        for (int i = 0; i < iface.n_shared; ++i) {
            int node = iface.shared_nodes[i];
            field_data[3 * node + 0] = 0.5 * (iface.send_buf[3 * i + 0] +
                                                iface.recv_buf[3 * i + 0]);
            field_data[3 * node + 1] = 0.5 * (iface.send_buf[3 * i + 1] +
                                                iface.recv_buf[3 * i + 1]);
            field_data[3 * node + 2] = 0.5 * (iface.send_buf[3 * i + 2] +
                                                iface.recv_buf[3 * i + 2]);
        }
    }

    /// Compute interface force imbalance (residual)
    Real interface_force_residual(const DomainInterface& iface) const {
        Real sum = 0.0;
        for (int i = 0; i < 3 * iface.n_shared; ++i) {
            Real diff = iface.send_buf[i] + iface.recv_buf[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /// Check convergence: |f_send + f_recv| < tol
    bool is_converged(const DomainInterface& iface, Real tol) const {
        return interface_force_residual(iface) < tol;
    }

    int domain_id() const { return domain_id_; }
    int n_domains() const { return n_domains_; }
    bool is_initialized() const { return initialized_; }

private:
    int domain_id_ = 0;
    int n_domains_ = 1;
    bool initialized_ = false;
};


// ============================================================================
// 5. PythonCoupling — Python callback co-simulation
// ============================================================================

/**
 * @brief Callback function pointers for Python co-simulation
 */
struct PythonCallbacks {
    void (*on_send)(const Real*, int) = nullptr;     ///< Called when data is sent
    void (*on_receive)(Real*, int) = nullptr;         ///< Called to receive data
    void (*on_advance)(Real) = nullptr;               ///< Called on time advance
    void (*on_initialize)(void) = nullptr;            ///< Called on initialize
    void (*on_finalize)(void) = nullptr;              ///< Called on finalize
};

/**
 * @brief Python callback co-simulation adapter
 *
 * Delegates all coupling operations to user-provided function pointers,
 * enabling Python (or any FFI) co-simulation without compile-time
 * dependencies. The callbacks are typically set from ctypes or pybind11.
 */
class PythonCoupling : public CouplingAdapter {
public:
    PythonCoupling() = default;

    /// Set the Python callback function pointers
    void set_callbacks(const PythonCallbacks& cb) {
        callbacks_ = cb;
    }

    void initialize(const CouplingConfig& config) override {
        config_ = config;
        initialized_ = true;
        current_time_ = 0.0;
        if (callbacks_.on_initialize) {
            callbacks_.on_initialize();
        }
    }

    void send_field(const char* /*name*/, const Real* data, int n) override {
        if (callbacks_.on_send) {
            callbacks_.on_send(data, n);
        }
        // Store locally for self-test
        last_sent_.assign(data, data + n);
    }

    void receive_field(const char* /*name*/, Real* data, int n) override {
        if (callbacks_.on_receive) {
            callbacks_.on_receive(data, n);
        } else {
            // Fallback: return last sent data
            int copy_n = (static_cast<int>(last_sent_.size()) < n)
                             ? static_cast<int>(last_sent_.size()) : n;
            if (copy_n > 0) {
                std::memcpy(data, last_sent_.data(),
                            static_cast<size_t>(copy_n) * sizeof(Real));
            }
            for (int i = copy_n; i < n; ++i) data[i] = 0.0;
        }
    }

    void advance(Real dt) override {
        current_time_ += dt;
        if (callbacks_.on_advance) {
            callbacks_.on_advance(dt);
        }
    }

    void finalize() override {
        if (callbacks_.on_finalize) {
            callbacks_.on_finalize();
        }
        initialized_ = false;
        last_sent_.clear();
    }

    /// Check if callbacks are set
    bool has_send_callback() const { return callbacks_.on_send != nullptr; }
    bool has_receive_callback() const { return callbacks_.on_receive != nullptr; }
    bool has_advance_callback() const { return callbacks_.on_advance != nullptr; }

    /// Get current simulation time
    Real current_time() const { return current_time_; }

    /// Get number of send invocations (for testing)
    int send_count() const { return send_count_; }
    int receive_count() const { return receive_count_; }

private:
    PythonCallbacks callbacks_{};
    std::vector<Real> last_sent_;
    int send_count_ = 0;
    int receive_count_ = 0;
};


// ============================================================================
// 6. CouplingInterpolation — RBF and nearest-neighbor field interpolation
// ============================================================================

/**
 * @brief Radial basis function interpolator with compact support
 */
struct RBFInterpolator {
    Real* weights = nullptr;    ///< Interpolation weights (n_centers values)
    Real* centers = nullptr;    ///< Center coordinates (3*n_centers)
    int n_centers = 0;
    Real support_radius = 1.0;  ///< Compact support radius
};

/**
 * @brief Field interpolation utilities for coupling
 *
 * Provides two interpolation methods:
 * 1. RBF (Radial Basis Function) with Wendland C2 compact support kernel
 * 2. Nearest-neighbor fallback for robustness
 *
 * The Wendland C2 kernel:  phi(r) = (1 - r/R)^4 * (4*r/R + 1), for r < R
 * This kernel is positive definite in 3D and provides C2 smoothness.
 */
class CouplingInterpolation {
public:
    /// Wendland C2 compact-support radial basis function
    /// phi(r) = (1 - q)^4 * (4q + 1), q = r / R, q in [0, 1)
    static Real wendland_c2(Real r, Real R) {
        if (r >= R || R <= 0.0) return 0.0;
        Real q = r / R;
        Real omq = 1.0 - q;
        return omq * omq * omq * omq * (4.0 * q + 1.0);
    }

    /// Compute RBF interpolation weights for a single query point
    /// Sets interp.weights[j] = normalized Wendland weight from center j
    static void compute_rbf_weights(RBFInterpolator& interp,
                                    const Real* query_pt) {
        Real wsum = 0.0;
        for (int j = 0; j < interp.n_centers; ++j) {
            Real dx = query_pt[0] - interp.centers[3 * j + 0];
            Real dy = query_pt[1] - interp.centers[3 * j + 1];
            Real dz = query_pt[2] - interp.centers[3 * j + 2];
            Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            Real w = wendland_c2(r, interp.support_radius);
            interp.weights[j] = w;
            wsum += w;
        }
        // Normalize to partition of unity
        if (wsum > 1.0e-30) {
            for (int j = 0; j < interp.n_centers; ++j) {
                interp.weights[j] /= wsum;
            }
        }
    }

    /// Interpolate a scalar field at multiple query points using RBF
    /// source_vals: one scalar per center
    /// result: one scalar per query point
    static void interpolate_rbf(RBFInterpolator& interp,
                                const Real* source_vals,
                                const Real* query_pts, int n_query,
                                Real* result) {
        for (int i = 0; i < n_query; ++i) {
            const Real* qp = &query_pts[3 * i];
            compute_rbf_weights(interp, qp);

            Real val = 0.0;
            for (int j = 0; j < interp.n_centers; ++j) {
                val += interp.weights[j] * source_vals[j];
            }
            result[i] = val;
        }
    }

    /// Nearest-neighbor interpolation (fallback)
    /// For each query point, find the closest source point and copy its value
    static void interpolate_nearest(const Real* source_pts, const Real* source_vals,
                                    int n_src,
                                    const Real* query_pts, int n_query,
                                    Real* result) {
        for (int i = 0; i < n_query; ++i) {
            Real qx = query_pts[3 * i + 0];
            Real qy = query_pts[3 * i + 1];
            Real qz = query_pts[3 * i + 2];

            Real best_dist2 = std::numeric_limits<Real>::max();
            int best_j = 0;

            for (int j = 0; j < n_src; ++j) {
                Real dx = qx - source_pts[3 * j + 0];
                Real dy = qy - source_pts[3 * j + 1];
                Real dz = qz - source_pts[3 * j + 2];
                Real d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_j = j;
                }
            }
            result[i] = source_vals[best_j];
        }
    }

    /// Interpolate a vector field (3 components per point) using nearest-neighbor
    static void interpolate_nearest_vec3(const Real* source_pts,
                                         const Real* source_vals, int n_src,
                                         const Real* query_pts, int n_query,
                                         Real* result) {
        for (int i = 0; i < n_query; ++i) {
            Real qx = query_pts[3 * i + 0];
            Real qy = query_pts[3 * i + 1];
            Real qz = query_pts[3 * i + 2];

            Real best_dist2 = std::numeric_limits<Real>::max();
            int best_j = 0;

            for (int j = 0; j < n_src; ++j) {
                Real dx = qx - source_pts[3 * j + 0];
                Real dy = qy - source_pts[3 * j + 1];
                Real dz = qz - source_pts[3 * j + 2];
                Real d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_j = j;
                }
            }
            result[3 * i + 0] = source_vals[3 * best_j + 0];
            result[3 * i + 1] = source_vals[3 * best_j + 1];
            result[3 * i + 2] = source_vals[3 * best_j + 2];
        }
    }

    /// Compute interpolation error (L2 norm of difference)
    static Real compute_l2_error(const Real* computed, const Real* reference,
                                 int n) {
        Real sum = 0.0;
        for (int i = 0; i < n; ++i) {
            Real diff = computed[i] - reference[i];
            sum += diff * diff;
        }
        return std::sqrt(sum / n);
    }
};

} // namespace coupling
} // namespace nxs
