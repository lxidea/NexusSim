(part2)=
# Part II — Core Framework

This part describes the foundational infrastructure upon which all NexusSim modules are
built: the type system, memory management, GPU abstraction layer, data structures, and
input/output subsystem.

---

(ch05_infrastructure)=
## Core Infrastructure

### Type System

The type system is defined in `core/types.hpp` and establishes the fundamental numeric
types used throughout the framework.

#### Precision Types

| Type    | C++ Definition   | Purpose                                |
|---------|------------------|----------------------------------------|
| `Real`  | `double`         | Floating-point precision (configurable via `NEXUSSIM_REAL`) |
| `Index` | `std::size_t`    | Array indices and sizes                |
| `Int`   | `std::int32_t`   | Integer values                         |
| `Int64` | `std::int64_t`   | Large integer values (step counters)   |

The precision of `Real` may be changed at compile time by defining the preprocessor
symbol `NEXUSSIM_REAL` (for example, `-DNEXUSSIM_REAL=float` for single-precision builds).

#### Fixed-Size Vector Types

NexusSim provides fixed-size vector aliases based on `std::array`:

| Type       | Definition            | Common Aliases       |
|------------|-----------------------|----------------------|
| `Vec2<T>`  | `std::array<T, 2>`   | `Vec2r`, `Vec2i`     |
| `Vec3<T>`  | `std::array<T, 3>`   | `Vec3r`, `Vec3i`     |
| `Vec4<T>`  | `std::array<T, 4>`   | `Vec4r`, `Vec4i`     |
| `Vec6<T>`  | `std::array<T, 6>`   | `Vec6r` (Voigt tensors) |

#### Enumerations

The following scoped enumerations define the principal categorization types:

**`FieldType`** — `Scalar`, `Vector`, `Tensor`

**`FieldLocation`** — `Node`, `Element`, `IntegrationPoint`, `Particle`

**`SolverType`** — `Explicit`, `Implicit`, `QuasiStatic`

**`TimeIntegrator`** — `CentralDifference`, `NewmarkBeta`, `GeneralizedAlpha`, `RungeKutta4`

**`LinearSolverType`** — `Direct`, `ConjugateGradient`, `GMRES`, `BiCGSTAB`

**`ExecutionSpace`** — `CPU`, `GPU`, `Auto`

#### Mathematical Constants

The namespace `nxs::constants` provides compile-time mathematical constants as variable
templates:

| Constant          | Value                    |
|-------------------|--------------------------|
| `pi<T>`           | $3.14159265358979\ldots$ |
| `two_pi<T>`       | $2\pi$                   |
| `half_pi<T>`      | $\pi/2$                  |
| `sqrt_two<T>`     | $\sqrt{2}$               |
| `sqrt_three<T>`   | $\sqrt{3}$               |
| `epsilon<T>`      | $10^{-12}$ (double) or $10^{-6}$ (float) |

### Exception System

The exception hierarchy is defined in `core/exception.hpp`. All NexusSim exceptions
derive from the base class `nxs::Exception`, which itself inherits from
`std::runtime_error`. Each exception automatically captures the source location (file,
line, function) via C++20 `std::source_location` when available, or compiler built-ins
as a fallback.

| Exception Class         | Purpose                                    |
|-------------------------|--------------------------------------------|
| `Exception`             | Base class for all NexusSim exceptions      |
| `LogicError`            | Programming errors (assertion failures)     |
| `RuntimeError`          | Runtime errors                              |
| `InvalidArgumentError`  | Invalid function arguments                  |
| `OutOfRangeError`       | Index out of bounds                         |
| `NotImplementedError`   | Unimplemented features                      |
| `FileIOError`           | File operation failures                     |
| `ConvergenceError`      | Solver convergence failures                 |
| `GPUError`              | GPU/Kokkos errors                           |
| `MPIError`              | MPI communication errors                    |

#### Assertion Macros

The following macros provide guarded assertions that throw specific exception types:

```cpp
NXS_ASSERT(condition, message)        // Throws LogicError
NXS_REQUIRE(condition, message)       // Throws InvalidArgumentError
NXS_CHECK_RANGE(index, size)          // Throws OutOfRangeError
NXS_NOT_IMPLEMENTED()                 // Throws NotImplementedError
NXS_DEBUG_ASSERT(condition, message)  // Active only in debug builds
```

### GPU and Kokkos Abstraction Layer

The GPU abstraction layer, defined in `core/gpu.hpp`, provides a portable interface to
Kokkos. When Kokkos is not available (`NEXUSSIM_HAVE_KOKKOS` is not defined), all
GPU-related types and functions degrade gracefully to no-op stubs.

#### KokkosManager

The `KokkosManager` is a singleton responsible for Kokkos lifecycle management:

```cpp
class KokkosManager {
public:
    static KokkosManager& instance();
    void initialize(const KokkosConfig& config = KokkosConfig{});
    void finalize();
    bool is_initialized() const;
    void print_configuration() const;
};
```

**Parameters.** The `KokkosConfig` structure accepts:

- `num_threads` — Number of host threads (`-1` for auto-detection).
- `device_id` — GPU device identifier (default: `0`).
- `disable_warnings` — Suppress Kokkos warnings.
- `tune_internals` — Enable Kokkos internal tuning.

#### View Type Aliases

Kokkos views are aliased for convenience:

| Alias            | Kokkos Type                                          |
|------------------|------------------------------------------------------|
| `View1D<T>`      | `Kokkos::View<T*, DefaultMemSpace>`                  |
| `View2D<T>`      | `Kokkos::View<T**, LayoutLeft, DefaultMemSpace>`     |
| `View3D<T>`      | `Kokkos::View<T***, LayoutLeft, DefaultMemSpace>`    |
| `HostView1D<T>`  | `Kokkos::View<T*, HostMemSpace>`                     |
| `HostView2D<T>`  | `Kokkos::View<T**, LayoutLeft, HostMemSpace>`        |

#### Parallel Execution Wrappers

```cpp
template<typename Functor>
void parallel_for(const std::string& label, std::size_t n, const Functor& f);

template<typename Functor, typename ReduceType>
void parallel_reduce(const std::string& label, std::size_t n,
                     const Functor& f, ReduceType& result);

void fence(const std::string& label = "NexusSim::fence");
```

### Logging

The logging subsystem (`core/logger.hpp`) provides a singleton `Logger` with six severity
levels: `Trace`, `Debug`, `Info`, `Warn`, `Error`, `Critical`.

Logging macros follow the `fmt` library syntax:

```cpp
NXS_LOG_INFO("Mesh loaded: {} nodes, {} elements", num_nodes, num_elements);
NXS_LOG_WARN("CG solver: slow convergence at iteration {}", iter);
```

When spdlog is available (`NEXUSSIM_HAVE_SPDLOG`), the logger delegates to spdlog;
otherwise, it uses a lightweight built-in backend that writes to `stderr`.

### Memory Management

The memory subsystem (`core/memory.hpp`) provides four components:

#### Alignment Utilities

The `nxs::memory` namespace provides aligned allocation functions:

- `DefaultAlignment` — 64 bytes (cache-line alignment).
- `GPUAlignment` — 256 bytes (GPU memory alignment).
- `allocate_aligned(size, alignment)` — Allocates aligned memory.
- `free_aligned(ptr)` — Frees aligned memory.
- `is_aligned(ptr, alignment)` — Tests pointer alignment.

#### MemoryArena

A fast, stack-like allocator for temporary allocations:

```cpp
class MemoryArena {
public:
    explicit MemoryArena(std::size_t block_size = 1024 * 1024);
    void* allocate(std::size_t size, std::size_t alignment = 64);
    template<typename T, typename... Args>
    T* construct(Args&&... args);
    void reset();
    std::size_t total_allocated() const;
    std::size_t used() const;
};
```

The arena allocates memory in contiguous blocks (default: 1 MB). The `reset()` method
rewinds the allocation pointer without freeing memory, making it suitable for per-timestep
scratch allocations.

#### MemoryPool

A fixed-size object pool for rapid allocation and deallocation of objects of a single type:

```cpp
template<typename T, std::size_t ChunkSize = 1024>
class MemoryPool {
public:
    T* allocate();
    void deallocate(T* ptr);
    template<typename... Args>
    T* construct(Args&&... args);
    void destroy(T* ptr);
};
```

#### AlignedBuffer

An RAII wrapper for cache- or GPU-aligned memory buffers:

```cpp
template<typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t size, std::size_t alignment = 64);
    T* data();
    std::size_t size() const;
    T& operator[](std::size_t i);
    T& at(std::size_t i);         // Bounds-checked
    void resize(std::size_t new_size);
    void fill(const T& value);
    void zero();
};
```

#### MemoryTracker

A global singleton that records allocation statistics:

```cpp
class MemoryTracker {
public:
    static MemoryTracker& instance();
    void record_allocation(std::size_t size);
    void record_deallocation(std::size_t size);
    const MemoryStats& stats() const;  // total_allocated, peak_usage, etc.
};
```

### MPI Wrapper

The `MPIManager` singleton (`core/mpi.hpp`) provides a simplified interface to MPI
collective operations. When MPI is disabled (`NEXUSSIM_HAVE_MPI` is not defined), the
manager operates in serial mode (rank 0, size 1) and all collective operations reduce
to identity operations.

Supported collective operations include `broadcast`, `reduce_sum`, `allreduce_sum`,
`allreduce_max`, and `allreduce_min`.

---

(ch06_data)=
## Data Structures

This chapter describes the three principal data containers: `Field`, `Mesh`, and `State`.
These containers form the foundation of all NexusSim simulations.

### Field Container

The `Field<T>` class template (`data/field.hpp`) is the fundamental data container in
NexusSim. It stores simulation data — scalars, vectors, or tensors — at specified
locations (nodes, elements, integration points, or particles).

#### Storage Layout

Fields employ a **Structure-of-Arrays (SOA)** layout. For a vector field with three
components ($x$, $y$, $z$) and $n$ entities, the data are stored as:

$$
[\underbrace{x_0, x_1, \ldots, x_{n-1}}_{\text{component 0}},
 \underbrace{y_0, y_1, \ldots, y_{n-1}}_{\text{component 1}},
 \underbrace{z_0, z_1, \ldots, z_{n-1}}_{\text{component 2}}]
$$

This layout maximizes vectorization efficiency and GPU memory coalescing.

#### Construction

```cpp
Field(std::string name, FieldType type, FieldLocation location,
      std::size_t num_entities, std::size_t num_components = 1);
```

**Parameters:**
- `name` — Human-readable identifier for the field.
- `type` — One of `Scalar`, `Vector`, or `Tensor`.
- `location` — One of `Node`, `Element`, `IntegrationPoint`, or `Particle`.
- `num_entities` — Number of entities (nodes, elements, etc.).
- `num_components` — Number of components per entity.

**Validation:** The constructor enforces that scalar fields have exactly 1 component,
vector fields have 2–4 components, and tensor fields have 6 (Voigt) or 9 (full)
components.

#### Access Patterns

**Scalar access** (single-component fields):

```cpp
T& operator[](std::size_t i);
```

**Multi-component access:**

```cpp
T& at(std::size_t entity_id, std::size_t comp_id);
```

**Vector access:**

```cpp
template<std::size_t N>
std::array<T, N> get_vec(std::size_t entity_id) const;

template<std::size_t N>
void set_vec(std::size_t entity_id, const std::array<T, N>& vec);
```

**Component span** (access all values for one component):

```cpp
std::span<T> component(std::size_t comp);
```

#### Factory Functions

Three convenience functions simplify field creation:

```cpp
Field<T> make_scalar_field(name, location, num_entities);
Field<T> make_vector_field(name, location, num_entities, dim = 3);
Field<T> make_tensor_field(name, location, num_entities, use_voigt = true);
```

#### Statistical Operations

The `Field` class provides `min()`, `max()`, `sum()`, and `mean()` methods for
computing aggregate statistics over all stored values.

### Mesh

The `Mesh` class (`data/mesh.hpp`) represents the geometric and topological discretization
of the computational domain.

#### Node Data

The mesh stores node coordinates as a three-component vector field. The following
operations are available:

```cpp
std::size_t num_nodes() const;
void set_node_coordinates(std::size_t node_id, const Vec3r& coords);
Vec3r get_node_coordinates(std::size_t node_id) const;
Field<Real>& coordinates();
```

#### Element Blocks

Elements are organized into **element blocks**, where each block contains a homogeneous
group of elements of the same type.

```cpp
struct ElementBlock {
    std::string name;
    ElementType type;
    std::size_t num_nodes_per_elem;
    std::size_t num_elements() const;
    std::span<Index> element_nodes(std::size_t elem_id);
};
```

**Adding an element block:**

```cpp
Index add_element_block(const std::string& name, ElementType type,
                        std::size_t num_elems, std::size_t nodes_per_elem);
```

**Returns:** The block identifier, which may be used to retrieve the block via
`element_block(block_id)`.

#### Node Sets and Side Sets

Named collections of nodes and element faces are supported:

```cpp
void add_node_set(const std::string& name, const std::vector<Index>& node_ids);
void add_side_set(const std::string& name,
                  const std::vector<std::pair<Index, Index>>& elem_side_pairs);
```

#### Bounding Box

```cpp
struct BoundingBox {
    Vec3r min, max;
    Real volume() const;
    Vec3r center() const;
};

BoundingBox compute_bounding_box() const;
```

### Simulation State

The `State` class (`data/state.hpp`) manages all time-dependent simulation fields and
provides a unified interface for accessing and updating them.

#### Construction

The `State` object is constructed with a reference to a `Mesh` and automatically
allocates five default nodal fields:

| Field Name       | Type   | Components | Description                 |
|------------------|--------|------------|-----------------------------|
| `displacement`   | Vector | 3          | Nodal displacements         |
| `velocity`       | Vector | 3          | Nodal velocities            |
| `acceleration`   | Vector | 3          | Nodal accelerations         |
| `force`          | Vector | 3          | Nodal force resultants      |
| `mass`           | Scalar | 1          | Lumped nodal masses         |

#### Time Management

```cpp
Real time() const;
void advance_time(Real dt);
Int64 step() const;
void advance_step();
```

#### Field Access

Convenience accessors are provided for standard fields:

```cpp
Field<Real>& displacement();
Field<Real>& velocity();
Field<Real>& acceleration();
Field<Real>& force();
Field<Real>& mass();
Field<Real>& stress();    // Optional — must be added explicitly
Field<Real>& strain();    // Optional — must be added explicitly
```

Additional fields may be registered dynamically:

```cpp
void add_field(const std::string& name, Field<Real>&& field);
bool has_field(const std::string& name) const;
Field<Real>& field(const std::string& name);
```

#### Energy Computation

```cpp
Real compute_kinetic_energy() const;
// Returns: 0.5 * sum_i(m_i * |v_i|^2)
```

### State History

The `StateHistory` class manages multiple time levels for multi-step time integration
schemes. It maintains a circular buffer of `State` objects:

```cpp
class StateHistory {
public:
    explicit StateHistory(const Mesh& mesh, std::size_t num_levels = 2);
    State& current();
    State& previous(std::size_t n = 1);
    void advance();
};
```

---

(ch07_io)=
## Input/Output

The I/O subsystem comprises header files in the `io/` directory, organized into
input readers, output writers, and checkpoint/restart facilities. With the addition of
production output formats (Wave 20), multi-format readers (Wave 22), and preprocessing
utilities (Wave 22), the I/O directory has grown to approximately 20 header files.

### Input Readers

#### LS-DYNA Keyword Reader

The `LSDynaReader` class (`io/lsdyna_reader.hpp`) parses LS-DYNA keyword-format (`.k`)
files. The reader supports approximately 30 keywords organized into eight groups:

| Group        | Supported Keywords                                       |
|--------------|----------------------------------------------------------|
| Nodes        | `*NODE`                                                  |
| Elements     | `*ELEMENT_SOLID`, `*ELEMENT_SHELL`, `*ELEMENT_BEAM`     |
| Sections     | `*SECTION_SOLID`, `*SECTION_SHELL`, `*SECTION_BEAM`, `*SECTION_SPH` |
| Materials    | `*MAT_ELASTIC`, `*MAT_PLASTIC_KINEMATIC`, `*MAT_JOHNSON_COOK`, and others |
| Contacts     | `*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE`, `*CONTACT_TIED` |
| Loads        | `*LOAD_NODE`, `*LOAD_BODY`, `*BOUNDARY_PRESCRIBED_MOTION` |
| Curves       | `*DEFINE_CURVE`                                          |
| Constraints  | `*CONSTRAINED_NODAL_RIGID_BODY`, `*CONSTRAINED_EXTRA_NODES` |

The reader has been validated with 172 test assertions (see {ref}`Chapter 25 <ch25_testing>`).

#### Radioss Reader

The `RadiossReader` class (`io/radioss_reader.hpp`) parses Radioss legacy card-based
Starter deck format.

#### Configuration Reader

The `ConfigReader` class (`io/config_reader.hpp`) parses YAML configuration files for
simulation parameters and material definitions. This reader requires the yaml-cpp
library.

#### Mesh Validator

The `MeshValidator` class (`io/mesh_validator.hpp`) performs quality checks on imported
meshes, including connectivity validation, element orientation checks, and duplicate
node detection.

### Output Writers

#### VTK Writer

The `VTKWriter` class (`io/vtk_writer.hpp`) writes simulation results in VTK format for
visualization in ParaView or similar tools:

```cpp
class VTKWriter {
public:
    void write_mesh(const std::string& filename, const Mesh& mesh);
    void write_field(const std::string& filename, const Mesh& mesh,
                     const Field<Real>& field);
    void write_time_step(const std::string& base, int step,
                         const Mesh& mesh, const State& state);
};
```

Both legacy (`.vtk`) and XML (`.vtu`) formats are supported.

#### Animation Writer

The `AnimationWriter` class (`io/animation_writer.hpp`) manages time-series output for
creating animation sequences. It writes VTK files at user-specified intervals and
generates a ParaView Data (`.pvd`) collection file.

### Checkpoint and Restart

#### Basic Checkpoint

The `CheckpointWriter` and `CheckpointReader` classes (`io/checkpoint.hpp`) provide
binary checkpoint I/O:

```cpp
class CheckpointWriter {
    void write(const std::string& filename, const State& state, const Mesh& mesh);
};

class CheckpointReader {
    State read(const std::string& filename, const Mesh& mesh);
    bool validate_header(const std::string& filename);
};
```

The binary format consists of a header (magic number, version, endianness indicator)
followed by serialized state data.

#### Extended Checkpoint

The `ExtendedCheckpointWriter` and `ExtendedCheckpointReader` classes
(`io/extended_checkpoint.hpp`) extend the basic checkpoint with:

- Material history variables (all 32 per integration point).
- Contact state (active pairs, penetration depths).
- Failure flags (per element).
- Sensor data (complete time histories).

### Enhanced Output Modules

| Module                 | Header                          | Description                        |
|------------------------|---------------------------------|------------------------------------|
| Time History           | `io/time_history.hpp`           | Nodal/element/global results vs. time |
| Part Energy            | `io/part_energy.hpp`            | KE, IE, contact energy per part    |
| Interface Force        | `io/interface_force_output.hpp` | Contact forces at interfaces       |
| Cross-Section Force    | `io/cross_section_force.hpp`    | Section cuts: $F_x, F_y, F_z, M_x, M_y, M_z$ |
| Result Database        | `io/result_database.hpp`        | Binary D3PLOT-like format          |
