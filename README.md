# NexusSim - Next-Generation Computational Mechanics Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25+-blue.svg)](https://cmake.org/)

Modern, GPU-accelerated computational mechanics solver for multi-physics simulations at exascale.

---

## Project Context

> **For complete project context, ecosystem details, and session recovery info, see [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)**

### What This Project Is

NexusSim is part of a larger **multi-physics simulation platform** initiative:

```
Parent Project: /mnt/d/_working_/FEM-PD/
├── OpenRadioss/       → Legacy Fortran FEM (reference implementation)
├── PeriSys-Haoran/    → CUDA Peridynamics solver (fracture/fragmentation)
├── claude-radioss/    → THIS PROJECT - NexusSim unified C++20/Kokkos framework
└── docs/              → Cross-project specifications
```

### Vision

1. **Migrate from OpenRadioss** - Port critical FEM functionality to modern C++20/Kokkos with GPU acceleration
2. **Integrate Peridynamics** - Couple with PeriSys for crack propagation and fragmentation
3. **Unified Multi-Physics** - FEM + SPH + Peridynamics + DEM in one framework
4. **Exascale Performance** - 80% efficiency at 10K cores, 5x+ GPU speedup

### Current Status (2026-02-25)

```
Wave 0: Foundation           [████████████████████] 100% ✅
Wave 1: Preprocessing/Mesh   [███████████████░░░░░]  75% ✅
Wave 2: Explicit Solver      [████████████████████] 100% ✅
Wave 3: Implicit Solver      [████████████████████] 100% ✅
Wave 4: Multi-Physics        [████████████████████] 100% ✅
Gap Waves 1-8: Feature Closure [████████████████████] 100% ✅
```

---

## Features (Updated 2026-02-25)

**Core FEM**:
- ✅ **10 Element Types**: Hex8, Hex20, Tet4, Tet10, Shell3, Shell4, Wedge6, Beam2, Truss, Spring/Damper
- ✅ **GPU Acceleration**: Kokkos parallel kernels (298M DOFs/sec on OpenMP)
- ✅ **Explicit Dynamics**: Central difference, adaptive timestep, subcycling
- ✅ **14+ Material Models**: Elastic, Von Mises, Johnson-Cook, Neo-Hookean, Mooney-Rivlin, Ogden, orthotropic, foam, crushable foam, honeycomb, viscoelastic, Cowper-Symonds, Zhao, piecewise-linear, tabulated, rigid, null
- ✅ **6 Failure Models**: Hashin, Tsai-Wu, Chang-Chang, GTN, GISSMO, tabulated
- ✅ **Contact**: Penalty, node-to-surface, mortar, Hertzian, tied (with failure), Coulomb friction
- ✅ **Element Erosion**: 15+ failure criteria, mass redistribution

**Implicit Solver (Complete)**:
- ✅ **Static & Dynamic**: FEMStaticSolver, FEMImplicitDynamicSolver (Newmark-β)
- ✅ **Linear Solvers**: CG (Jacobi preconditioned), Direct LU
- ✅ **Newton-Raphson**: Line search, load stepping
- ✅ **Arc-Length Method**: Crisfield's cylindrical method for snap-through/buckling, adaptive step sizing
- ✅ **PETSc Integration**: Optional scalable backend (CG/GMRES/LU/AMG), behind compile guard
- ✅ **Shell4 6-DOF**: Auto-detection, local→global transform, mixed mesh support

**Multi-Physics**:
- ✅ **SPH Solver**: Weakly compressible, multiple kernels (Cubic, Wendland, Quintic)
- ✅ **FEM-SPH Coupling**: Penalty + pressure coupling for FSI
- ✅ **Thermal Coupling**: Conduction, thermo-mechanical effects
- ✅ **Peridynamics**: Bond-based, state-based, correspondence PD with FEM coupling
- ✅ **Composite Laminates**: CLT, thermal residual stress, progressive failure, strength envelopes

**Infrastructure**:
- ✅ **Modern C++20**: Kokkos for GPU portability
- ✅ **I/O**: VTK, Radioss, LS-DYNA readers, checkpoint/restart, enhanced output
- ✅ **Sensors & Controls**: 5 sensor types with CFC filtering, 8 control actions
- ✅ **ALE**: 3 smoothing methods, 2 advection methods
- ✅ **5 EOS Models**: Ideal Gas, Gruneisen, JWL, polynomial, tabulated

## Quick Start

### Prerequisites

- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.25+
- Conan 2.0 or vcpkg
- MPI implementation (OpenMPI 4.1+, MPICH, Intel MPI)
- Optional: CUDA 11.8+ or ROCm 5.4+ for GPU support

### Build from Source

```bash
# Clone repository
git clone https://github.com/nexussim/nexussim.git
cd nexussim

# Install dependencies with Conan
conan install . --output-folder=build --build=missing

# Configure and build
cmake --preset conan-release
cmake --build --preset conan-release

# Run tests
ctest --preset conan-release
```

### Using vcpkg

```bash
# Install dependencies
vcpkg install eigen3 spdlog hdf5 catch2 pybind11

# Configure with vcpkg toolchain
cmake -B build -S . \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

## Example Usage

### C++ API

```cpp
#include <nexussim/nexussim.hpp>

int main() {
    // Create mesh
    auto mesh = nxs::Mesh::from_file("model.msh");

    // Define material
    auto steel = nxs::Material::elastic(210e9, 0.3, 7850);
    mesh.assign_material("part1", steel);

    // Create simulation
    nxs::Simulation sim(mesh);
    sim.set_solver(nxs::SolverType::Explicit);
    sim.set_end_time(0.1);

    // Run
    sim.run();

    return 0;
}
```

### Python API

```python
import nexussim as nxs

# Create mesh
mesh = nxs.Mesh.from_file("model.msh")

# Define material (Johnson-Cook)
aluminum = nxs.Material.johnson_cook(
    A=324e6, B=114e6, n=0.42, C=0.002, m=1.34
)
mesh.assign_material("hood", aluminum)

# Simulation setup
sim = nxs.Simulation(mesh)
sim.set_solver(type="explicit", end_time=0.1)

# Multi-GPU configuration
sim.set_parallel(mpi_ranks=4, gpus_per_rank=2)

# Run
sim.run()

# Post-process
results = nxs.Results("output.h5")
max_stress = results.field("stress").max()
print(f"Max stress: {max_stress/1e6:.1f} MPa")
```

## Project Structure

```
NexusSim/
├── include/           # Public headers
│   └── nexussim/
├── src/               # Implementation
│   ├── core/          # Infrastructure (memory, threading, GPU, logging)
│   ├── data/          # Data structures (Mesh, Field, State)
│   ├── discretization/# FEM, meshfree, coupling
│   ├── materials/     # Material models
│   ├── solvers/       # Time integration, linear/nonlinear solvers
│   ├── physics/       # Physics modules (solid, fluid, thermal, EM)
│   ├── contact/       # Contact mechanics
│   ├── io/            # Input/output (HDF5, VTK, Exodus)
│   └── tests/         # Unit and integration tests
├── python/            # Python bindings
├── docs/              # Documentation and specifications
└── examples/          # Example programs
```

## Documentation

**Quick Start**:
- [Current TODO](TODO.md) - What to work on now
- [Development Reference](DEVELOPMENT_REFERENCE.md) - Feature planning guide
- [Documentation Map](DOCUMENTATION_MAP.md) - Complete navigation

**Detailed Documentation** (see [docs/README.md](docs/README.md) for complete index):
- [Architecture Design](docs/Unified_Architecture_Blueprint.md)
- [Progress Analysis](docs/PROGRESS_VS_GOALS_ANALYSIS.md)
- [Element Library Status](docs/ELEMENT_LIBRARY_STATUS.md)
- [Known Issues](docs/KNOWN_ISSUES.md)
- [Coupling Specification](docs/Coupling_GPU_Specification.md)
- [Migration Roadmap](docs/Legacy_Migration_Roadmap.md)
- [API Reference](https://nexussim.readthedocs.io) (Coming soon)

## Roadmap (Updated 2026-02-25)

### All Major Development Waves Complete

**Wave 0: Foundation** ✅ **100%**
- [x] C++20/Kokkos project skeleton, data containers, build system, YAML, VTK

**Wave 1: Preprocessing & Mesh** ✅ **75%**
- [x] Mesh ingestion, programmatic creation, Radioss/LS-DYNA readers

**Wave 2: Explicit Solver** ✅ **100%**
- [x] 10 element types, GPU kernels, materials, contact, erosion, adaptive timestep

**Phase 3A-C: Advanced Physics** ✅ **100%**
- [x] Thermal coupling, SPH solver, FEM-SPH coupling, energy monitoring

**Wave 3: Implicit Solver** ✅ **100%**
- [x] Sparse matrix (CSR), CG/Direct solvers, Newton-Raphson with line search
- [x] Newmark-β integrator, FEM static & implicit dynamic solvers
- [x] Shell4 6-DOF integration (auto-detection, local→global transform)
- [x] Arc-length method (Crisfield's cylindrical, adaptive step sizing)
- [x] PETSc integration (optional, CG/GMRES/LU/AMG, behind compile guard)

**Wave 4: Peridynamics** ✅ **100%**
- [x] Bond-based, state-based, correspondence PD, FEM-PD coupling, element morphing

**Gap Waves 1-8: Feature Closure** ✅ **100%**
- [x] 14 material models, 6 failure models, rigid bodies/constraints, loads, EOS
- [x] Checkpoint/restart, enhanced output, composites, sensors/controls, ALE

### Remaining Work

- [ ] GPU benchmarks (requires NVIDIA GPU hardware)
- [ ] Full MPI-parallel solver integration
- [ ] Automatic mesh refinement

## Performance Targets

| Problem Size | Hardware | Expected Performance |
|--------------|----------|---------------------|
| 100K nodes | 1 GPU | 50-100x vs CPU |
| 1M nodes | 8 GPUs | Linear scaling |
| 10M nodes | 64 GPUs | 80% scaling efficiency |
| 100M nodes | 1024 GPUs | Exascale-ready |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use NexusSim in your research, please cite:

```bibtex
@software{nexussim2025,
  title = {NexusSim: Next-Generation Computational Mechanics Framework},
  author = {NexusSim Development Team},
  year = {2025},
  url = {https://github.com/nexussim/nexussim}
}
```

## Contact

- GitHub Issues: https://github.com/nexussim/nexussim/issues
- Documentation: https://nexussim.readthedocs.io
- Email: dev@nexussim.org
