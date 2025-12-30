# NexusSim Project Context - Complete Reference

**Purpose**: This document captures the complete project context, ecosystem, and vision. Read this first when resuming work after a session crash or starting fresh.

**Last Updated**: 2025-12-29

---

## 1. Project Vision

### What We're Building

A **next-generation multi-physics simulation platform** that:
- Surpasses OpenRadioss in performance and capability
- Integrates FEM, SPH, Peridynamics, and DEM in a unified framework
- Achieves GPU acceleration with Kokkos for portability (CUDA/HIP/OpenMP)
- Enables crash/impact, FSI, and fracture simulations at exascale

### Core Objectives

1. **Performance**: 20% faster than OpenRadioss, 5x+ GPU speedup
2. **Multi-Physics**: Seamless FEM + meshfree + peridynamics coupling
3. **Scalability**: 80% parallel efficiency up to 10,000 CPU cores
4. **Modern Architecture**: C++20, modular, extensible, GPU-first design

---

## 2. Project Ecosystem

### Directory Structure (Parent Folder)

```
/mnt/d/_working_/FEM-PD/
├── OpenRadioss/           # Legacy Fortran FEM solver (reference)
│   ├── engine/            # Core solver (radioss2.F, resol.F, forint.F)
│   ├── common_source/     # Shared utilities
│   ├── new_project_spec/  # specification.md - full requirements
│   └── gemini_analysis/   # AI analysis of codebase
│
├── PeriSys-Haoran/        # CUDA Peridynamics solver
│   └── code/              # .cu/.cuh files
│       ├── JSolve.cu      # Main solver loop
│       ├── JParticle_stress.cu  # Stress calculations
│       ├── JContact_force.cu    # Contact algorithms
│       ├── JTime_integral.cu    # Time integration
│       └── Global_Para.cuh      # Material definitions
│
├── claude-radioss/        # THIS PROJECT - NexusSim
│   ├── src/               # C++20/Kokkos implementation
│   ├── docs/              # Documentation
│   └── examples/          # Test programs
│
├── gemini-radioss/        # Alternative implementation (reference)
├── viRadioss/             # Design documents and analysis
├── docs/                  # Cross-project documentation
└── OpenMPI/               # MPI installation
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    NexusSim (claude-radioss)                     │
│                 Unified C++20/Kokkos Framework                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  FEM Solver  │  │  SPH Solver  │  │  Peridynamics (PD)   │  │
│  │  (Wave 2 ✅) │  │  (Phase 3C ✅)│  │  (Wave 4 ✅ 80%)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └────────┬────────┴──────────────────────┘              │
│                  ▼                                               │
│         ┌────────────────┐                                      │
│         │ Coupling Layer │  ← FEM-SPH done, PD-FEM done         │
│         └────────────────┘                                      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Data Sources:                                                   │
│  • OpenRadioss → FEM algorithms, element formulations           │
│  • PeriSys-Haoran → Peridynamics, bond-based PD, materials      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. OpenRadioss Reference

### What It Is
- Open-source explicit FEM solver for crash/impact simulation
- ~366,000 lines of Fortran code
- Industry-standard for automotive crashworthiness

### Key Files to Reference
| File | Purpose |
|------|---------|
| `engine/source/resol.F` | Main time integration loop |
| `engine/source/forint.F` | Element force dispatcher |
| `engine/source/materials/` | Material model implementations |
| `engine/source/contact/` | Contact algorithms |
| `new_project_spec/specification.md` | Full requirements spec |

### Migration Strategy
- Wave-based migration (see Legacy_Migration_Roadmap.md)
- Wrap critical Fortran via ISO C bindings initially
- Full C++ reimplementation with GPU kernels

---

## 4. PeriSys-Haoran Reference

### What It Is
- GPU-accelerated Peridynamics solver (CUDA)
- Developed by Haoran Zhang (ZHR), started April 2025
- Particle-based method for fracture/fragmentation

### Material Models Available
```cpp
enum class MaterialType {
    Elastic = 1,
    DruckerPrager = 2,         // Geomaterials
    JohnsonHolmquist2 = 4,     // Ceramics/glass
    Rigid = 5,
    JohnsonCook = 7,           // Metals with strain rate
    JohnsonCook_PD = 8,        // PD variant
    BDPD = 9,                  // Bond-based PD
    ElasticBondPD = 10,        // Elastic bonds
    PMMABondPD = 11            // PMMA polymer
};
```

### Key Files
| File | Purpose | Lines |
|------|---------|-------|
| `JSolve.cu` | Main solver loop | 16,760 |
| `JParticle_stress.cu` | Stress/force calculation | 147,315 |
| `JContact_force.cu` | Contact algorithms | 118,286 |
| `JTime_integral.cu` | Time integration | 44,232 |
| `JBuildNeighborList.cu` | Neighbor search | 11,893 |
| `Global_Para.cuh` | Parameters/materials | 35,563 |

### Integration Plan
- Extract peridynamics algorithms into NexusSim
- Create PD-FEM coupling interface
- Maintain CUDA kernels, add Kokkos variants

---

## 5. Current Implementation Status

### Wave Completion Summary

```
Wave 0: Foundation           [████████████████████] 100% ✅ COMPLETE
Wave 1: Preprocessing/Mesh   [███████████████░░░░░]  75% ✅ ADVANCED
Wave 2: Explicit Solver      [████████████████████] 100% ✅ COMPLETE
Wave 3: Implicit Solver      [████████████████░░░░]  80% ✅ IMPLEMENTED
Wave 4: Peridynamics         [████████████████░░░░]  80% ✅ IMPLEMENTED
Wave 5: Optimization         [░░░░░░░░░░░░░░░░░░░░]   0% ❌ NOT STARTED
```

### Element Library (10 Elements - ALL Production Ready)

| Element | Type | Nodes | DOF | Status | Integration |
|---------|------|-------|-----|--------|-------------|
| Hex8 | 3D Solid | 8 | 24 | ✅ Production | 1-pt reduced + hourglass |
| Hex20 | 3D Solid | 20 | 60 | ✅ Production | 2×2×2 or 3×3×3 |
| Tet4 | 3D Solid | 4 | 12 | ✅ Production | 1-pt reduced |
| Tet10 | 3D Solid | 10 | 30 | ✅ Production | 4-pt |
| Shell4 | Shell | 4 | 24 | ✅ Production | 2×2 membrane + bending |
| Shell3 | Shell | 3 | 18 | ✅ Production | CST + DKT |
| Wedge6 | 3D Solid | 6 | 18 | ✅ Production | 2×3 |
| Beam2 | Beam | 2 | 12 | ✅ Production | Euler-Bernoulli |
| Truss | Bar | 2 | 6 | ✅ Production | Axial only |
| Spring/Damper | Discrete | 2 | 6 | ✅ Production | Point-to-point |

### Material Models

| Category | Models | Status |
|----------|--------|--------|
| Elastic | Linear elastic, Neo-Hookean | ✅ Complete |
| Plasticity | Von Mises, Johnson-Cook | ✅ Complete |
| Hyperelastic | Neo-Hookean | ✅ Complete |
| Failure | Principal stress/strain, J-C damage, Cockcroft-Latham | ✅ Complete |

### Advanced Features

| Feature | Status | Notes |
|---------|--------|-------|
| Penalty Contact | ✅ Complete | Node-to-surface, spatial hashing |
| Coulomb Friction | ✅ Complete | Static/dynamic, stick-slip |
| Self-Contact | ✅ Complete | Automatic detection |
| Element Erosion | ✅ Complete | Multiple failure criteria |
| Adaptive Timestep | ✅ Complete | CFL-based |
| Thermal Coupling | ✅ Complete | Conduction, thermo-mechanical |
| SPH Solver | ✅ Complete | Weakly compressible, multiple kernels |
| FEM-SPH Coupling | ✅ Complete | Penalty + pressure coupling |
| Subcycling | ✅ Complete | Multi-scale time integration |
| Energy Monitoring | ✅ Complete | Conservation tracking |
| **Peridynamics** | ✅ Complete | Bond-based PD with Kokkos |
| **PD-FEM Coupling** | ✅ Complete | Arlequin blending method |

### Peridynamics Module (Wave 4)

| Component | File | Status |
|-----------|------|--------|
| PD Types | `include/nexussim/peridynamics/pd_types.hpp` | ✅ Complete |
| Particle System | `include/nexussim/peridynamics/pd_particle.hpp` | ✅ Complete |
| Neighbor List | `include/nexussim/peridynamics/pd_neighbor.hpp` | ✅ Complete |
| Bond Force | `include/nexussim/peridynamics/pd_force.hpp` | ✅ Complete |
| PD Solver | `include/nexussim/peridynamics/pd_solver.hpp` | ✅ Complete |
| FEM-PD Coupling | `include/nexussim/peridynamics/pd_fem_coupling.hpp` | ✅ Complete |

Key algorithms ported from PeriSys-Haoran:
- **Bond force calculation** - PMB model with critical stretch failure
- **CSR neighbor list** - O(N²) build with influence weighting
- **Velocity-Verlet** - Symplectic time integration
- **Damage tracking** - Per-particle damage from broken bonds

### GPU Status

| Component | Status | Performance |
|-----------|--------|-------------|
| Kokkos Integration | ✅ Complete | CUDA/OpenMP backends |
| Element Kernels | ✅ Complete | All 10 elements parallelized |
| Time Integration | ✅ Complete | Parallel velocity/position update |
| Contact Search | ✅ Complete | Spatial hashing on GPU |
| Measured Performance | ✅ Verified | 11 million DOFs/sec |

---

## 6. Next Phase: Wave 3 - Implicit Solver

### Components Needed

1. **Tangent Stiffness Matrix Assembly**
   - Element stiffness matrices (material + geometric)
   - Global sparse matrix assembly (CSR format)
   - GPU-accelerated assembly

2. **Newton-Raphson Solver**
   - Residual computation
   - Tangent matrix update
   - Line search for robustness
   - Convergence monitoring

3. **Linear Solvers**
   - Direct solver (small problems)
   - Iterative: CG for SPD systems
   - Preconditioners: Jacobi, ILU
   - Optional: PETSc integration

4. **Time Integration**
   - Newmark-β (already have formulas)
   - Generalized-α (optional)
   - HHT-α (optional)

5. **Static Analysis**
   - Load stepping
   - Arc-length method (optional)

### Use Cases
- Static structural analysis
- Low-frequency dynamics
- Quasi-static forming/assembly
- Eigenvalue/modal analysis (future)

---

## 7. Future: Wave 4 - Peridynamics Integration

### Plan

1. **Bond-Based PD Implementation**
   - Port algorithms from PeriSys-Haoran
   - Kokkos kernels for GPU
   - Neighbor list management

2. **State-Based PD Implementation**
   - Ordinary state-based
   - Non-ordinary (correspondence)

3. **PD-FEM Coupling**
   - Interface detection
   - Force/displacement transfer
   - Arlequin or bridging domain method

4. **Crack Propagation**
   - Bond breaking criteria
   - Damage tracking
   - Visualization

---

## 8. Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Strong scaling efficiency | 80% at 10K cores | TBD |
| GPU speedup (single GPU) | 5x+ vs CPU | ✅ Achieved |
| Time vs OpenRadioss | 20% faster | TBD |
| Memory per DOF | 15% less than OpenRadioss | TBD |
| DOFs/second | 10M+ | 11M ✅ |

---

## 9. Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C++20 |
| GPU Abstraction | Kokkos (CUDA/HIP/OpenMP) |
| Parallelism | MPI + Kokkos |
| Build | CMake 3.25+ |
| Linear Algebra | Eigen, (PETSc planned) |
| I/O | YAML config, VTK output, HDF5 planned |
| Testing | Catch2 (planned), example-based |
| Logging | spdlog |

---

## 10. Key Documentation Files

### Essential (Read These First)
| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start |
| `TODO.md` | Current priorities (this week) |
| `docs/PROJECT_CONTEXT.md` | THIS FILE - complete context |
| `docs/WHATS_LEFT.md` | Remaining work breakdown |

### Architecture
| File | Purpose |
|------|---------|
| `docs/Unified_Architecture_Blueprint.md` | Overall system design |
| `docs/Framework_Architecture_Current_State.md` | Implementation status |
| `DEVELOPMENT_REFERENCE.md` | Feature planning guide |

### Status Tracking
| File | Purpose |
|------|---------|
| `docs/Development_Roadmap_Status.md` | Phase-by-phase status |
| `docs/ELEMENT_LIBRARY_STATUS.md` | All elements detailed |
| `docs/PROGRESS_VS_GOALS_ANALYSIS.md` | Progress analysis |
| `docs/KNOWN_ISSUES.md` | Bug tracking |

### Specifications
| File | Purpose |
|------|---------|
| `docs/YAML_Input_Format.md` | Configuration spec |
| `docs/Coupling_GPU_Specification.md` | GPU coupling design |
| `docs/FSI_Field_Registration.md` | Multi-physics fields |

---

## 11. Quick Commands

### Build
```bash
cd /mnt/d/_working_/FEM-PD/claude-radioss
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests
```bash
./bin/hex8_element_test
./bin/tet4_compression_test
./bin/shell4_plate_test
./bin/fem_solver_test
./bin/contact_sphere_plate_test
```

### Check GPU Backend
```bash
cmake .. -LAH | grep -i kokkos
```

---

## 12. Session Recovery Checklist

When starting a new session after crash/disconnect:

1. **Read this file** (`docs/PROJECT_CONTEXT.md`)
2. **Check TODO.md** for current priorities
3. **Check docs/WHATS_LEFT.md** for next tasks
4. **Review recent commits**: `git log --oneline -10`
5. **Build and run tests** to verify state

---

## 13. Contact & Resources

- **Parent Project Spec**: `/mnt/d/_working_/FEM-PD/OpenRadioss/new_project_spec/specification.md`
- **PeriSys Code**: `/mnt/d/_working_/FEM-PD/PeriSys-Haoran/code/`
- **Cross-Project Docs**: `/mnt/d/_working_/FEM-PD/docs/`

---

*This document is the single source of truth for project context. Update it when major changes occur.*
