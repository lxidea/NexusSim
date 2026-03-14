(part1)=
# Part I — User Guide

This part introduces the NexusSim framework, describes how to obtain, build, and verify the
software, and provides a guided tutorial for first-time users.

---

(ch01_introduction)=
## Introduction

### Purpose

NexusSim is a next-generation computational mechanics framework designed for multi-physics
simulations at exascale. It provides a unified platform encompassing the Finite Element
Method (FEM), Smoothed Particle Hydrodynamics (SPH), and Peridynamics (PD), with hardware
portability achieved through the Kokkos programming model.

The framework is written in C++20 and adopts a header-heavy architecture to facilitate
template inlining and GPU kernel generation. It targets both shared-memory parallelism
(OpenMP) and GPU acceleration (CUDA, HIP, SYCL) through a single, portable source base.

### Capabilities

NexusSim provides the following principal capabilities:

1. **Structural analysis** — Linear and nonlinear static and dynamic analyses using ten
   element types, fourteen material models, and six failure criteria.
2. **Explicit dynamics** — Central-difference time integration with adaptive time stepping,
   subcycling, element erosion, and energy monitoring.
3. **Implicit solvers** — Newton–Raphson iteration with conjugate gradient and direct
   linear solvers, Newmark-$\beta$ time integration, and arc-length path tracing.
4. **Contact mechanics** — Seven contact formulations including penalty, mortar, Hertzian,
   and tied contact with failure.
5. **Smoothed Particle Hydrodynamics** — Weakly compressible SPH with multiple kernel
   functions and neighbor search algorithms.
6. **Peridynamics** — Bond-based, ordinary state-based, and non-ordinary correspondence
   formulations with specialized ceramic, metal, and geomaterial models.
7. **Multi-physics coupling** — FEM–SPH and FEM–PD coupling via Arlequin blending, mortar
   integrals, element morphing, and adaptive zone reclassification.
8. **Composite analysis** — Classical Lamination Theory with thermal residual stresses,
   interlaminar shear, and progressive failure analysis.
9. **ALE mesh management** — Lagrangian–Eulerian mesh smoothing and advection.
10. **Sensors and controls** — Runtime monitoring and simulation steering.

### Codebase Metrics

The following table summarizes the current state of the codebase:

| Metric               | Value       |
|----------------------|-------------|
| Header files (.hpp)  | 126         |
| Source files (.cpp)   | 15          |
| Test files (.cpp)     | 95          |
| Header lines of code | ~46,100     |
| Source lines of code  | ~4,700      |
| Test lines of code    | ~36,100     |
| Total lines of code   | ~86,900     |
| Element types         | 10          |
| Material models (FEM) | 14          |
| Material models (PD)  | 3           |
| Failure models        | 6           |
| EOS models            | 5           |
| Contact formulations  | 7           |
| Test assertions       | 1,400+      |

### Technology Stack

| Component         | Technology                                              |
|-------------------|---------------------------------------------------------|
| Language          | C++20 (ISO/IEC 14882:2020)                              |
| GPU portability   | Kokkos 3.7                                              |
| Parallelism       | OpenMP (default), CUDA (available via Kokkos rebuild)   |
| Build system      | CMake 3.25+                                             |
| Optional packages | MPI, PETSc, HDF5, Eigen3, spdlog, yaml-cpp             |

---

(ch02_installation)=
## Installation and Build

### Prerequisites

The following software is required to build NexusSim:

- A C++20-compliant compiler (GCC 11+, Clang 14+, or MSVC 19.30+)
- CMake 3.25 or later
- Kokkos 3.7 (bundled in the `external/` directory)

The following packages are optional:

| Package   | Purpose                        | CMake Option                 |
|-----------|--------------------------------|------------------------------|
| MPI       | Distributed-memory parallelism | `NEXUSSIM_ENABLE_MPI`        |
| PETSc     | Scalable linear solvers        | `NEXUSSIM_ENABLE_PETSC`      |
| HDF5      | High-performance I/O           | Detected automatically       |
| Eigen3    | Dense linear algebra           | Detected automatically       |
| spdlog    | Structured logging backend     | Detected automatically       |
| yaml-cpp  | YAML configuration parsing     | Detected automatically       |
| pybind11  | Python bindings                | `NEXUSSIM_BUILD_PYTHON`      |

### Building from Source

The standard build procedure is as follows:

```bash
# Configure the build (minimal configuration)
cmake -S . -B build \
  -DNEXUSSIM_ENABLE_MPI=OFF \
  -DNEXUSSIM_BUILD_PYTHON=OFF

# Compile
cmake --build build -j$(nproc)
```

### CMake Options

The following table lists all user-configurable CMake options:

| Option                     | Default | Description                         |
|----------------------------|---------|-------------------------------------|
| `NEXUSSIM_ENABLE_GPU`      | `ON`    | Enable GPU acceleration via Kokkos  |
| `NEXUSSIM_ENABLE_MPI`      | `ON`    | Enable MPI distributed parallelism  |
| `NEXUSSIM_BUILD_PYTHON`    | `ON`    | Build Python bindings via pybind11  |
| `NEXUSSIM_ENABLE_PETSC`    | `OFF`   | Enable PETSc solver backend         |

### Compile Definitions

The build system defines the following preprocessor symbols when the corresponding
packages are detected:

| Definition               | Condition         | Effect                         |
|--------------------------|-------------------|--------------------------------|
| `NEXUSSIM_HAVE_MPI`      | MPI found         | Enables MPI code paths         |
| `NEXUSSIM_HAVE_KOKKOS`   | Kokkos found      | Enables GPU code paths         |
| `NEXUSSIM_HAVE_PETSC`    | PETSc found       | Enables PETSc solvers          |
| `NEXUSSIM_HAVE_HDF5`     | HDF5 found        | Enables HDF5 I/O              |
| `NEXUSSIM_HAVE_SPDLOG`   | spdlog found      | Enables spdlog logging backend |

### Build Verification

After a successful build, the test suite may be executed to verify correctness:

```bash
# Run a single test
./build/bin/material_models_test

# Run all tests
for test in build/bin/*_test; do
    echo "Running $test..."
    "$test" || echo "FAILED: $test"
done
```

All tests should report zero failures. See {ref}`Chapter 25 <ch25_testing>` for a detailed
description of the test infrastructure.

### GPU Build Configuration

The development system is equipped with an NVIDIA GeForce GT 1030 GPU (Pascal
architecture, compute capability 6.1, 2 GB GDDR5, 384 CUDA cores). Three Kokkos
installations are available:

| Installation  | Location                             | Backends       | Compiler     |
|---------------|--------------------------------------|----------------|--------------|
| System        | `/usr/lib/cmake/Kokkos`              | OpenMP, Serial | g++          |
| CUDA (system) | `/usr/local/lib/cmake/Kokkos`        | CUDA, Serial   | nvcc_wrapper |
| CUDA (bundled)| `external/kokkos/lib/cmake/Kokkos`   | CUDA, Serial   | nvcc_wrapper |

The default build uses the system Kokkos (OpenMP backend). To build with CUDA
acceleration using the bundled Kokkos, specify the `nvcc_wrapper` compiler and set
`CUDA_ROOT` to match the CUDA toolkit version Kokkos was built with:

```bash
CUDA_ROOT=/usr/local/cuda-12.6 cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=$PWD/external/kokkos/bin/nvcc_wrapper \
  -DKokkos_DIR=$PWD/external/kokkos/lib/cmake/Kokkos \
  -DNEXUSSIM_ENABLE_GPU=ON \
  -DNEXUSSIM_ENABLE_MPI=OFF \
  -DNEXUSSIM_BUILD_PYTHON=OFF \
  -DNEXUSSIM_ENABLE_OPENMP=OFF \
  -DCMAKE_CXX_STANDARD=17

CUDA_ROOT=/usr/local/cuda-12.6 cmake --build build-cuda -j$(nproc)
```

When `NEXUSSIM_ENABLE_GPU` is `ON` and a CUDA-enabled Kokkos is found, the build
system defines `NEXUSSIM_HAVE_KOKKOS` and links against Kokkos CUDA runtime libraries.
All Kokkos-annotated kernels (`KOKKOS_INLINE_FUNCTION`) will then execute on the GPU.

#### GPU Performance (GT 1030)

The following table summarizes GPU speedup over the 64-thread OpenMP CPU backend on
representative benchmarks.  The GT 1030 is an entry-level GPU; production compute GPUs
would show substantially larger speedups.

| Benchmark                   | Problem Size    | CPU (ms) | GPU (ms) | Speedup |
|-----------------------------|-----------------|----------|----------|---------|
| Vector Add (DAXPY)          | 100K            | 4.32     | 0.19     | 22.5x   |
| Element Forces (Hex8)       | 12.5K elements  | 9.08     | 0.18     | 49.6x   |
| CG Solve (3D Laplacian)     | 27K DOFs        | 2955.7   | 118.5    | 24.9x   |
| FEM Explicit Step (Hex8)    | 27K elements    | 13.53    | 2.72     | 5.0x    |

The GPU excels at compute-bound operations (element forces, CG solver) while
memory-bandwidth-bound operations (SpMV at 1M, PD bond forces) favor the CPU on
this entry-level hardware.  See {ref}`Chapter 25 <ch25_testing>` for detailed results.

### Platform Notes

**Linux (GCC 13.3):** This is the primary development platform. The default build uses
the OpenMP backend for shared-memory parallelism on the host CPU. CUDA acceleration
requires the NVIDIA CUDA Toolkit 12.0 or later, a compute-capable GPU device, and a
Kokkos installation built with the CUDA backend and `nvcc_wrapper`.

**macOS:** Supported via Clang. OpenMP support requires `libomp` from Homebrew. CUDA is
not supported on macOS.

**Windows:** Supported via MSVC 2022 or later. The build has been tested with the Visual
Studio CMake generator.

---

(ch03_quickstart)=
## Quick Start Tutorial

This chapter provides three minimal working examples that demonstrate the principal
capabilities of NexusSim. Each example is self-contained and can be compiled as a
standalone executable.

### Example 1: Creating a Mesh

The following program creates a simple hexahedral mesh, assigns node coordinates, and
writes the mesh to VTK format for visualization.

```cpp
#include <nexussim/data/mesh.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <iostream>

int main() {
    // Create a mesh with 8 nodes (one hex element)
    nxs::Mesh mesh(8);

    // Define node coordinates for a unit cube
    mesh.set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh.set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh.set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh.set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh.set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh.set_node_coordinates(5, {1.0, 0.0, 1.0});
    mesh.set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh.set_node_coordinates(7, {0.0, 1.0, 1.0});

    // Add one Hex8 element block
    auto block_id = mesh.add_element_block("solid", nxs::ElementType::Hex8, 1, 8);
    auto& block = mesh.element_block(block_id);
    auto nodes = block.element_nodes(0);
    for (size_t i = 0; i < 8; ++i) nodes[i] = i;

    mesh.print_info();
    return 0;
}
```

### Example 2: Linear Static FEM Analysis

The following program performs a linear static analysis of a cantilever beam modeled with
hexahedral elements. One end is fully constrained; a point load is applied to the opposite
end.

```cpp
#include <nexussim/data/mesh.hpp>
#include <nexussim/solver/fem_static_solver.hpp>
#include <iostream>

int main() {
    // Build a 10-element cantilever beam mesh
    const int nx = 10;
    const int num_nodes = (nx + 1) * 4;  // 4 nodes per cross-section
    nxs::Mesh mesh(num_nodes);

    // ... (assign coordinates and connectivity) ...

    // Create the static solver
    nxs::solver::FEMStaticSolver solver(mesh);
    solver.set_material(210.0e9, 0.3);  // Steel: E = 210 GPa, nu = 0.3

    // Fix the left end (nodes 0–3)
    for (int i = 0; i < 4; ++i)
        solver.fix_node_all(i);

    // Apply downward force at the right end
    solver.add_force(num_nodes - 1, 2, -1000.0);  // Fz = -1000 N

    // Solve
    solver.solve_linear();

    // Report maximum displacement
    std::cout << "Max displacement: " << solver.max_displacement() << " m\n";
    return 0;
}
```

### Example 3: Explicit Dynamics with Contact

The following example illustrates how to set up an explicit dynamics simulation with
contact detection. The reader is referred to {ref}`Chapter 13 <ch13_explicit>` and
{ref}`Chapter 16 <ch16_contact>` for detailed descriptions of the solver and contact
algorithms.

```cpp
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/fem/contact.hpp>
#include <nexussim/data/state.hpp>

int main() {
    nxs::Mesh mesh(/* ... */);
    nxs::State state(mesh);

    // Configure the explicit solver
    nxs::physics::AdaptiveTimestep timestep;
    Real dt = timestep.compute_stable_dt(mesh, state);

    // Time loop
    for (int step = 0; step < 1000; ++step) {
        // 1. Compute internal forces
        // 2. Detect and enforce contact
        // 3. Time-integrate (central difference)
        state.advance_time(dt);
        state.advance_step();
    }
    return 0;
}
```

---

(ch04_architecture)=
## Architecture Overview

### Directory Structure

The NexusSim source tree is organized as follows:

```text
NexusSim/
├── include/nexussim/          126 public headers
│   ├── core/          (7)     Type system, exceptions, GPU, logging, memory, MPI
│   ├── data/          (4)     Mesh, Field, State containers
│   ├── coupling/      (3)     Multi-physics coupling operators
│   ├── discretization/(11)    Element implementations
│   ├── solver/        (6)     Implicit solvers, sparse matrix
│   ├── physics/       (43)    Materials, failure, composites, EOS, time integration
│   ├── fem/           (18)    Contact, constraints, loads, sensors, controls
│   ├── io/            (14)    Readers, writers, checkpoint, output
│   ├── sph/           (4)     SPH solver, kernels, coupling
│   ├── peridynamics/  (15)    PD solver, materials, coupling
│   ├── ale/           (1)     ALE mesh management
│   └── utils/         (1)     Performance instrumentation
├── src/                       15 source files
├── examples/                  95 test executables
├── cmake/                     CMake find-modules
└── docs/                      Documentation
```

### Design Philosophy

The framework is built on the following design principles:

1. **Header-heavy architecture.** The majority of implementation resides in header files
   to enable template instantiation and `KOKKOS_INLINE_FUNCTION` GPU kernel generation.
   Only platform-specific or large translation units are placed in `.cpp` files.

2. **Kokkos portability.** All performance-critical data structures use Kokkos views, and
   computational kernels are annotated with `KOKKOS_INLINE_FUNCTION` so that the same
   source code compiles for CPU (OpenMP, Serial) and GPU (CUDA, HIP, SYCL) backends.

3. **Structure-of-Arrays (SOA) layout.** The `Field<T>` container stores each component
   contiguously (all $x$-components, then all $y$-components, etc.) to maximize
   vectorization efficiency and cache utilization.

4. **Zero-overhead abstractions.** The type system, smart pointers, and container wrappers
   are designed to compile down to the same machine code as hand-written C equivalents.

### Namespace Organization

All NexusSim symbols reside within the `nxs` namespace:

```text
nxs::                              Root namespace
├── physics::                      Material models, elements, time integration, EOS, ALE
│   ├── MaterialType               Material type enumeration (14 types)
│   ├── ElementType                Element type enumeration (12 types)
│   ├── failure::                  Failure models
│   └── composite::                Composite analysis
├── solver::                       Implicit solvers, sparse matrices
├── coupling::                     Multi-physics coupling operators
├── pd::                           Peridynamics
├── sph::                          Smoothed Particle Hydrodynamics
├── io::                           Input/output subsystem
├── memory::                       Memory alignment and allocation
├── utils::                        Performance instrumentation
└── constants::                    Mathematical and physical constants
```

### Design Patterns

The framework employs the following design patterns:

**Singleton pattern.** Used for global services that require exactly one instance:
`Logger`, `Profiler`, `FieldRegistry`, `MPIManager`, `KokkosManager`, `MemoryTracker`.

**Factory pattern.** `ElementFactory` creates element objects by type;
`CouplingOperatorFactory` creates coupling operators by method.

**RAII (Resource Acquisition Is Initialization).** `KokkosManager` initializes Kokkos in
its constructor and finalizes it in its destructor. `ScopedTimer`, `MemoryArena`, and
`PETScContext` follow the same pattern.

**Callback architecture.** The Newton–Raphson and arc-length solvers accept
`std::function` callbacks for internal force computation and tangent stiffness assembly,
decoupling the solver from the element library.

### Data Flow

A typical explicit dynamics simulation follows this data flow:

1. **Mesh creation** — Nodes and element connectivity are defined in the `Mesh` object.
2. **State initialization** — The `State` object allocates default nodal fields
   (displacement, velocity, acceleration, force, mass).
3. **Material assignment** — `MaterialProperties` and `MaterialState` are assigned to
   each element.
4. **Time loop** — At each step:
   a. Element internal forces are computed from current displacements and stresses.
   b. Contact forces are detected and enforced.
   c. The time integrator advances the solution (central difference).
   d. Boundary conditions are applied.
   e. Output is written at specified intervals.
