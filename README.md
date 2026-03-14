# NexusSim — Next-Generation Computational Mechanics Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25+-blue.svg)](https://cmake.org/)

NexusSim is a modern, GPU-portable computational mechanics framework for multi-physics
simulations. It provides a unified platform for the Finite Element Method (FEM), Smoothed
Particle Hydrodynamics (SPH), and Peridynamics (PD), with hardware portability through the
Kokkos programming model.

## Features

**Finite Elements** — 10 element types (Hex8, Hex20, Tet4, Tet10, Wedge6, Shell4, Shell3,
Beam2, Truss, Spring/Damper), 14 material models, 6 failure criteria.

**Solvers** — Explicit dynamics (central difference, adaptive timestep, subcycling),
implicit statics and dynamics (Newton-Raphson, Newmark-beta, arc-length method),
optional PETSc backend.

**Contact** — Penalty, node-to-surface, surface-to-surface, mortar, Hertzian, tied
contact with failure, Coulomb friction.

**Multi-Physics** — Weakly compressible SPH, bond-based / state-based / correspondence
peridynamics, FEM-SPH and FEM-PD coupling (Arlequin, mortar, morphing, adaptive),
thermal analysis, ALE mesh management.

**Composites** — Classical Lamination Theory, ABD matrix, thermal residual stress,
interlaminar shear, progressive failure analysis.

**I/O** — LS-DYNA and Radioss readers, VTK writer, checkpoint/restart, time history,
animation output.

**Sensors and Controls** — 5 sensor types with CFC filtering (SAE J211), 8 control
action types for runtime simulation steering.

## Quick Start

### Prerequisites

- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.25+
- Kokkos 3.7+ (bundled in `external/`)

### Build

```bash
cmake -S . -B build \
  -DNEXUSSIM_ENABLE_MPI=OFF \
  -DNEXUSSIM_BUILD_PYTHON=OFF

cmake --build build -j$(nproc)
```

### Build with CUDA

To enable GPU acceleration, build against a CUDA-enabled Kokkos installation:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/nvcc_wrapper \
  -DKokkos_DIR=/usr/local/lib/cmake/Kokkos \
  -DNEXUSSIM_ENABLE_GPU=ON \
  -DNEXUSSIM_ENABLE_MPI=OFF \
  -DNEXUSSIM_BUILD_PYTHON=OFF

cmake --build build-cuda -j$(nproc)
```

### Run Tests

```bash
./build/bin/material_models_test
./build/bin/implicit_validation_test
./build/bin/failure_models_test
```

## Example

```cpp
#include <nexussim/data/mesh.hpp>
#include <nexussim/solver/fem_static_solver.hpp>

int main() {
    // Create a mesh
    nxs::Mesh mesh(44);
    // ... define nodes and element connectivity ...

    // Set up the static solver
    nxs::solver::FEMStaticSolver solver(mesh);
    solver.set_material(210.0e9, 0.3);  // Steel

    // Apply boundary conditions
    for (int i = 0; i < 4; ++i)
        solver.fix_node_all(i);
    solver.add_force(43, 2, -1000.0);

    // Solve and report
    solver.solve_linear();
    std::cout << "Max displacement: " << solver.max_displacement() << " m\n";
    return 0;
}
```

## Project Structure

```
NexusSim/
├── include/nexussim/       Public headers (126 files)
│   ├── core/               Type system, exceptions, GPU, logging, memory, MPI
│   ├── data/               Mesh, Field, State containers
│   ├── discretization/     Element implementations
│   ├── solver/             Implicit solvers, sparse matrix
│   ├── physics/            Materials, failure, composites, EOS, time integration
│   ├── fem/                Contact, constraints, loads, sensors, controls
│   ├── io/                 Readers, writers, checkpoint, output
│   ├── sph/                SPH solver, kernels, coupling
│   ├── peridynamics/       PD solver, materials, coupling
│   └── coupling/           Multi-physics coupling operators
├── src/                    Source files
├── examples/               Test executables
├── docs/manual/            Sphinx documentation (HTML + PDF)
└── cmake/                  CMake find-modules
```

## Documentation

The full software manual is built with Sphinx and available in both HTML and PDF:

```bash
cd docs/manual
make html    # HTML output in _build/html/
make pdf     # PDF output in docs/NexusSim_Software_Manual.pdf
make all     # Both
```

Requires: `pip install sphinx sphinx-book-theme myst-parser`

## CMake Options

| Option                  | Default | Description                        |
|-------------------------|---------|------------------------------------|
| `NEXUSSIM_ENABLE_GPU`   | `ON`    | Enable GPU acceleration via Kokkos |
| `NEXUSSIM_ENABLE_MPI`   | `ON`    | Enable MPI distributed parallelism |
| `NEXUSSIM_BUILD_PYTHON` | `ON`    | Build Python bindings (pybind11)   |
| `NEXUSSIM_ENABLE_PETSC` | `OFF`   | Enable PETSc solver backend        |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
