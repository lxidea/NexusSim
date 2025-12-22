# NexusSim - Next-Generation Computational Mechanics Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25+-blue.svg)](https://cmake.org/)

Modern, GPU-accelerated computational mechanics solver for multi-physics simulations at exascale.

## Features (Updated 2025-11-07)

**Currently Working**:
- ✅ **Solid Mechanics FEM**: 7 element types (Hex8, Hex20, Tet4, Tet10, Shell4, Wedge6, Beam2)
- ✅ **GPU Acceleration**: Kokkos parallel kernels implemented (80% complete)
- ✅ **Explicit Dynamics**: Central difference time integration (GPU-parallelized)
- ✅ **Modern C++20**: Clean architecture with Kokkos for GPU portability
- ✅ **VTK Output**: Full visualization support

**Planned Features**:
- ⚠️ **Multi-Physics**: Solid mechanics working, fluid/thermal/EM planned
- ⚠️ **FEM-Meshfree Coupling**: Architecture ready, implementation pending
- ⚠️ **Scalable**: Single-GPU ready, MPI + multi-GPU planned
- ⚠️ **Python API**: Planned for Phase 3

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

## Roadmap (Updated 2025-11-07)

**Phase 1 (Months 1-6)** - Foundation ✅ **85% COMPLETE**
- [x] Core infrastructure
- [x] 7 FEM element types (6 production-ready, 1 needs mesh fix)
- [x] Explicit time integration (GPU-parallelized)
- [ ] MPI parallelization

**Phase 2 (Months 7-12)** - GPU Acceleration ⚠️ **80% COMPLETE**
- [x] Kokkos integration
- [x] GPU element kernels (all 7 elements have parallel loops)
- [x] Time integration parallelized
- [ ] GPU backend verification (CUDA vs OpenMP)
- [ ] Multi-GPU support

**Phase 3 (Months 13-18)** - Advanced Features
- [ ] Implicit solver
- [ ] 100+ material models
- [ ] Meshfree methods (SPH, RKPM, PD)
- [ ] FEM-Meshfree coupling

**Phase 4 (Months 19-24)** - Multi-Physics
- [ ] FSI coupling
- [ ] Thermal-mechanical
- [ ] Electromagnetic

**Phase 5 (Months 25-30)** - Production
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Validation benchmarks
- [ ] Open-source release

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
