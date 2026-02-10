# NexusSim Development TODO

**Quick Reference**: Active development priorities
**Complete Context**: See `docs/PROJECT_CONTEXT.md` for full project ecosystem
**Last Updated**: 2026-02-10

---

## Current Status Summary

```
Wave 0: Foundation           [████████████████████] 100% ✅
Wave 1: Preprocessing/Mesh   [███████████████░░░░░]  75% ✅
Wave 2: Explicit Solver      [████████████████████] 100% ✅ COMPLETE!
Phase 3A-C: Advanced Physics [████████████████████] 100% ✅ COMPLETE!
Wave 3: Implicit Solver      [████████████████░░░░]  80% ✅ IMPLEMENTED!
Wave 4: Peridynamics         [████████████████████] 100% ✅ COMPLETE!
Wave 5: Optimization         [██████████████░░░░░░]  70% ⏳ GPU READY
Wave 6: FEM-PD Coupling      [████████████████████] 100% ✅ COMPLETE!
```

**Completed Features**:
- 10 element types (all production-ready)
- Von Mises + Johnson-Cook + Neo-Hookean materials
- Penalty contact with Coulomb friction
- Element erosion with multiple failure criteria
- SPH solver with FEM-SPH coupling
- Thermal coupling, subcycling, energy monitoring
- GPU acceleration ready (298M DOFs/sec on OpenMP, GPU pending)
- **Implicit solver infrastructure** (discovered 2025-12-28)
- **Peridynamics module** (implemented 2025-12-29)
- **Performance benchmark suite** (implemented 2025-12-29)
- **FEM-PD coupling infrastructure** (tested 2026-02-10, 38/38 tests passing)
- **FEM-PD integration test** (2026-02-10, 27/27 tests - domain setup, material consistency, blending)

---

## ✅ Wave 3 - Implicit Solver (MOSTLY IMPLEMENTED!)

### Implemented Components

The following were found already implemented during code review on 2025-12-28:

| Component | File | Status |
|-----------|------|--------|
| **Sparse Matrix (CSR)** | `include/nexussim/solver/implicit_solver.hpp` | ✅ Complete |
| **Sparse Matrix (CSR) v2** | `include/nexussim/solver/sparse_matrix.hpp` | ✅ Complete |
| **CG Solver** | `implicit_solver.hpp` (CGSolver) | ✅ Complete |
| **Direct Solver** | `implicit_solver.hpp` (DirectSolver) | ✅ Complete |
| **Jacobi Preconditioner** | `sparse_matrix.hpp` | ✅ Complete |
| **SSOR Preconditioner** | `sparse_matrix.hpp` | ✅ Complete |
| **Newton-Raphson** | `implicit_solver.hpp` (NewtonRaphsonSolver) | ✅ Complete |
| **Line Search** | Built into NewtonRaphsonSolver | ✅ Complete |
| **Newmark-β Integrator** | `implicit_solver.hpp` (NewmarkIntegrator) | ✅ Complete |
| **Static Solver** | `implicit_solver.hpp` (StaticSolver) | ✅ Complete |
| **FEM Static Solver** | `fem_static_solver.hpp` (FEMStaticSolver) | ✅ Complete |
| **FEM Implicit Dynamic** | `fem_static_solver.hpp` (FEMImplicitDynamicSolver) | ✅ Complete |
| **Element Stiffness** | Hex8, Tet4 stiffness matrices | ✅ Complete |
| **BC Application** | Penalty method with symmetry | ✅ Complete |
| **Stiffness Assembly** | `StiffnessAssembler` class | ✅ Complete |
| **Tests** | `examples/implicit_solver_test.cpp` | ✅ 10 test cases |

### Remaining Work (Wave 3)

- [ ] **Build and run tests** - Verify all tests pass
- [ ] **Add element stiffness** for Hex20, Tet10, shells
- [ ] **Validation** - Compare with analytical solutions
- [ ] **Arc-length method** (optional) - For snap-through buckling
- [ ] **PETSc integration** (optional) - For very large problems

---

## 🔥 CRITICAL: Next Priorities

### Priority 1: Verify Implicit Solver

1. Install cmake and build the project
2. Run `implicit_solver_test` and verify all tests pass
3. Run `implicit_newmark_test` and `implicit_dynamic_test`
4. Validate cantilever beam results against analytical solution

### Priority 2: Extend Element Support

1. Add `stiffness_matrix()` to Hex20Element
2. Add `stiffness_matrix()` to Tet10Element
3. Add `stiffness_matrix()` to Shell4Element
4. Integrate additional elements into FEMStaticSolver

### Priority 3: Wave 4 - Peridynamics Integration

See `docs/WHATS_LEFT.md` for Wave 4 details.

---

## ✅ Wave 4: Peridynamics Integration (80% IMPLEMENTED!)

### Implemented Components (2025-12-29)

| Component | File | Status |
|-----------|------|--------|
| **PD Types** | `include/nexussim/peridynamics/pd_types.hpp` | ✅ Complete |
| **PD Particle System** | `include/nexussim/peridynamics/pd_particle.hpp` | ✅ Complete |
| **PD Neighbor List (CSR)** | `include/nexussim/peridynamics/pd_neighbor.hpp` | ✅ Complete |
| **PD Bond Force** | `include/nexussim/peridynamics/pd_force.hpp` | ✅ Complete |
| **PD Solver** | `include/nexussim/peridynamics/pd_solver.hpp` | ✅ Complete |
| **FEM-PD Coupling** | `include/nexussim/peridynamics/pd_fem_coupling.hpp` | ✅ Complete |
| **PD Test Suite** | `examples/pd_bar_tension_test.cpp` | ✅ 5 test cases |

### Phase 4A: Bond-Based Peridynamics ✅ COMPLETE

- [x] **Port PeriSys algorithms** from `/mnt/d/_working_/FEM-PD/PeriSys-Haoran/code/`
  - ✅ `JParticle_stress.cu` → `pd_force.hpp` (PDBondForce)
  - ✅ `JBuildNeighborList.cu` → `pd_neighbor.hpp` (PDNeighborList)
  - ✅ `Global_Para.cuh` → `pd_types.hpp` (PDMaterial, PDMaterialType)

- [x] **Kokkos kernels** for GPU portability
- [x] **Material models**: Elastic with critical stretch failure

### Phase 4B: State-Based PD (Remaining)

- [ ] Ordinary state-based PD
- [ ] Non-ordinary (correspondence model)

### Phase 4C: PD-FEM Coupling ✅ COMPLETE

- [x] Interface detection (overlap region detection)
- [x] Arlequin/bridging domain method (blending functions)
- [x] Force/displacement transfer (FEMPDCoupling class)
- [x] CoupledFEMPDSolver for unified time stepping

### Completed Work (Wave 4) ✅

- [x] **Build and test** - All PD tests pass (11/11)
- [x] **State-based PD** - Ordinary state-based with dilatation
- [x] **Johnson-Cook** - With material presets (Al7075, Steel4340, Ti6Al4V, Copper)
- [x] **Drucker-Prager** - With presets (Sand, Clay, Concrete, Granite)
- [x] **Johnson-Holmquist 2** - With presets (Alumina, SiC, B4C, Glass)
- [x] **PD Contact** - Penalty-based with spatial hashing and friction

### Future Enhancements (Optional)

- [ ] Non-ordinary (correspondence) state-based PD
- [ ] Adaptive FEM-to-PD runtime conversion
- [ ] GPU-optimized contact with Kokkos

---

## ⏳ Wave 5: Performance Optimization (IN PROGRESS)

### Implemented Components (2025-12-29)

| Component | File | Status |
|-----------|------|--------|
| **Performance Timer** | `include/nexussim/utils/performance_timer.hpp` | ✅ Complete |
| **Kokkos Benchmark** | `examples/kokkos_performance_test.cpp` | ✅ Complete |
| **Comprehensive Benchmark** | `examples/comprehensive_benchmark.cpp` | ✅ Complete |
| **Memory Profiling** | Built into PerformanceStats | ✅ Complete |

### Phase 5A: Benchmarking Infrastructure ✅ COMPLETE

- [x] **Performance Timer Class** - High-resolution timing with `Timer`, `ScopedTimer`
- [x] **Statistics Collection** - Mean, std dev, min/max for multiple runs
- [x] **Global Profiler** - Named timers with `NXS_PROFILE_START/STOP` macros
- [x] **Memory Tracking** - Host/device memory usage in `MemoryStats`
- [x] **Benchmark Results** - Standardized `BenchmarkResult` struct

### Phase 5B: Kokkos Performance Tests ✅ COMPLETE

- [x] **Vector Operations** - DAXPY-style vector addition
- [x] **Reductions** - Dot product with `parallel_reduce`
- [x] **SpMV** - Sparse matrix-vector product
- [x] **Element Forces** - FEM-like gather-compute-scatter
- [x] **Time Integration** - DOF update kernel

### Benchmark Results (OpenMP Backend, 64 threads)

```
Peak Performance (3M DOFs):
  Element Processing: 10.4 million elements/sec
  DOF Update Rate:    298 million DOFs/sec
  Backend:            OpenMP (64 threads)
```

### Phase 5C: GPU Backend Support ✅ READY

| Backend | Status | Notes |
|---------|--------|-------|
| **OpenMP** | ✅ Active | 64 threads, 216M DOFs/sec |
| **CUDA** | ✅ Code ready | Waiting for NVIDIA GPU |
| **HIP/ROCm** | ✅ Code ready | Requires native Linux |

- [x] **Multi-backend support** - CUDA, HIP, OpenMP, Serial
- [x] **Backend detection** - Automatic at compile time
- [x] **GPU setup documentation** - See `docs/GPU_SETUP.md`
- [ ] **GPU benchmarks** - Run when GPU available
- [ ] **Memory Pool Allocator** - Reduce allocation overhead
- [ ] **Multi-GPU Scaling** - MPI + Kokkos for distributed GPU

### Performance Utilities

| Class | Purpose |
|-------|---------|
| `Timer` | High-resolution timing |
| `ScopedTimer` | RAII automatic timing |
| `PerformanceStats` | Statistical analysis |
| `Profiler` | Global named timers |
| `MemoryStats` | Memory usage tracking |
| `BenchmarkResult` | Standardized results |

### Usage Example

```cpp
#include <nexussim/utils/performance_timer.hpp>

// RAII-style timing
{
    nxs::utils::ScopedTimer timer("compute_forces");
    solver.compute_forces();
}  // Prints: [TIMER] compute_forces: 1.234 ms

// Manual timing
NXS_PROFILE_START("time_integration");
solver.step(dt);
NXS_PROFILE_STOP("time_integration");
NXS_PROFILE_PRINT();  // Print all stats
```

---

## 📊 Reference Information

### Implicit Solver Key Files

| File | Contents | Lines |
|------|----------|-------|
| `include/nexussim/solver/implicit_solver.hpp` | SparseMatrix, CGSolver, DirectSolver, NewtonRaphsonSolver, NewmarkIntegrator, StaticSolver | ~1000 |
| `include/nexussim/solver/sparse_matrix.hpp` | SparseMatrixCSR, JacobiPreconditioner, SSORPreconditioner, ConjugateGradientSolver, StiffnessAssembler, BoundaryConditionApplicator | ~670 |
| `include/nexussim/solver/fem_static_solver.hpp` | FEMStaticSolver, FEMImplicitDynamicSolver, mesh generators | ~1100 |
| `examples/implicit_solver_test.cpp` | 10 test cases | ~515 |
| `examples/implicit_newmark_test.cpp` | Newmark-β time integration tests | - |
| `examples/implicit_dynamic_test.cpp` | Dynamic implicit solver tests | - |

### Peridynamics Key Files (Wave 4)

| File | Contents | Lines |
|------|----------|-------|
| `include/nexussim/peridynamics/pd_types.hpp` | PDMaterial, PDMaterialType, Kokkos Views | ~275 |
| `include/nexussim/peridynamics/pd_particle.hpp` | PDParticleSystem, Velocity-Verlet | ~385 |
| `include/nexussim/peridynamics/pd_neighbor.hpp` | PDNeighborList (CSR), influence functions | ~300 |
| `include/nexussim/peridynamics/pd_force.hpp` | PDBondForce, bond failure | ~360 |
| `include/nexussim/peridynamics/pd_solver.hpp` | PDSolver, VTK output | ~300 |
| `include/nexussim/peridynamics/pd_fem_coupling.hpp` | FEMPDCoupling, CoupledFEMPDSolver | ~500 |
| `include/nexussim/peridynamics/pd_state_based.hpp` | PDStateForce, PDStateSolver, dilatation | ~450 |
| `include/nexussim/peridynamics/pd_materials.hpp` | Johnson-Cook, Drucker-Prager, JH-2, presets | ~500 |
| `include/nexussim/peridynamics/pd_contact.hpp` | PDContact, spatial hashing, friction | ~350 |
| `examples/pd_bar_tension_test.cpp` | 5 bond-based PD tests | ~300 |
| `examples/pd_validation_test.cpp` | 6 advanced PD tests | ~350 |

### Wave 5 Performance Files

| File | Contents | Lines |
|------|----------|-------|
| `include/nexussim/utils/performance_timer.hpp` | Timer, ScopedTimer, Profiler, MemoryStats, BenchmarkResult | ~280 |
| `examples/kokkos_performance_test.cpp` | Standalone Kokkos benchmarks | ~300 |
| `examples/comprehensive_benchmark.cpp` | FEM solver benchmarks | ~350 |
| `examples/gpu_performance_benchmark.cpp` | GPU-specific benchmarks | ~230 |

### PeriSys Reference (ported from)

| File | Purpose | Ported To |
|------|---------|-----------|
| `JParticle_stress.cu` | Stress calculation | `pd_force.hpp` |
| `JBuildNeighborList.cu` | Neighbor search | `pd_neighbor.hpp` |
| `Global_Para.cuh` | Materials | `pd_types.hpp` |
| `JSolve.cu` | Solver loop | `pd_solver.hpp` |

---

## 📚 Documentation Files

| Document | Purpose |
|----------|---------|
| `docs/PROJECT_CONTEXT.md` | **START HERE** - Complete project context |
| `docs/WHATS_LEFT.md` | Detailed remaining work |
| `docs/Development_Roadmap_Status.md` | Phase-by-phase status |
| `docs/ELEMENT_LIBRARY_STATUS.md` | Element details |

---

## Session Recovery Checklist

When starting after crash/disconnect:
1. Read `docs/PROJECT_CONTEXT.md` for full context
2. Check this file (`TODO.md`) for priorities
3. Run `git log --oneline -10` to see recent work
4. Build and run tests to verify state

---

*Last Updated: 2025-12-29*
*Current Focus: Wave 5 Performance Optimization (40% complete)*
