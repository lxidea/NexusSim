# What's Left - NexusSim Development Priorities

**Last Updated**: 2025-12-28
**Current Status**: Wave 3 MOSTLY IMPLEMENTED! Ready for verification and Wave 4 (Peridynamics)
**Complete Context**: See `PROJECT_CONTEXT.md` for full project ecosystem

---

## 🎉 Major Discovery (2025-12-28)

**Wave 3 Implicit Solver is ~80% implemented!** During code review, we discovered extensive implicit solver infrastructure already in place:

| Component | Status | Location |
|-----------|--------|----------|
| SparseMatrix (CSR) | ✅ DONE | `implicit_solver.hpp`, `sparse_matrix.hpp` |
| CG Solver | ✅ DONE | `implicit_solver.hpp` (CGSolver) |
| Direct Solver | ✅ DONE | `implicit_solver.hpp` (DirectSolver) |
| Preconditioners | ✅ DONE | `sparse_matrix.hpp` (Jacobi, SSOR) |
| Newton-Raphson | ✅ DONE | `implicit_solver.hpp` (NewtonRaphsonSolver) |
| Line Search | ✅ DONE | Built into Newton-Raphson |
| Newmark-β | ✅ DONE | `implicit_solver.hpp` (NewmarkIntegrator) |
| Static Solver | ✅ DONE | `implicit_solver.hpp` (StaticSolver) |
| FEM Static Solver | ✅ DONE | `fem_static_solver.hpp` (FEMStaticSolver) |
| FEM Implicit Dynamic | ✅ DONE | `fem_static_solver.hpp` (FEMImplicitDynamicSolver) |
| Test Suite | ✅ DONE | `examples/implicit_solver_test.cpp` (10 tests) |

**Remaining for Wave 3**:
- Build and run tests to verify
- Add element stiffness for Hex20, Tet10, shells
- Validate against analytical solutions
- (Optional) Arc-length method, PETSc integration

---

## Recent Completions (December 2025)

### Phase 3B: Advanced Time Integration ✅ COMPLETE

- [x] Subcycling for multi-scale problems (SubcyclingController)
- [x] Consistent mass matrix option (Sparse CSR with Jacobi solver)
- [x] Energy conservation monitoring (EnergyMonitor class)
- [x] Velocity-Verlet and Newmark-β integrators
- [x] 35/35 tests passing

### Phase 3C: Multi-Physics Foundation ✅ COMPLETE

**SPH Solver** 🌊
- [x] SPH particle discretization (SPHSolver class)
- [x] Neighbor search with spatial hashing (SpatialHashGrid, CompactNeighborList)
- [x] Kernel functions (Cubic Spline, Wendland C2/C4, Quintic Spline)
- [x] Weakly compressible formulation (Tait EOS)
- [x] Artificial viscosity and XSPH correction
- [x] Dam break simulation capability
- [x] 27/27 tests passing

**Fluid-Structure Interaction** 🔗
- [x] FEM-SPH coupling interface (FEMSPHCoupling class)
- [x] Surface extraction from FEM mesh (FEMSurface)
- [x] Penalty-based contact forces
- [x] Direct pressure coupling
- [x] Friction forces at interface
- [x] Newton's 3rd law force balance verified
- [x] CoupledFEMSPHSolver for staggered FSI
- [x] 16/16 tests passing

---

## ✅ Previously Completed

### Wave 2 Completion ✅

**Element Library - 10 Elements, ALL Production-Ready**:
| Element | Type | Nodes | Status |
|---------|------|-------|--------|
| Hex8 | 3D Solid | 8 | ✅ Full + reduced integration |
| Hex20 | 3D Quadratic Solid | 20 | ✅ 2×2×2 or 3×3×3 integration |
| Tet4 | 3D Solid | 4 | ✅ 1-point reduced |
| Tet10 | 3D Quadratic Solid | 10 | ✅ 4-point |
| Shell4 | Quadrilateral Shell | 4 | ✅ Membrane + bending |
| Shell3 | Triangular Shell | 3 | ✅ CST + DKT |
| Wedge6 | Prism | 6 | ✅ 2×3 integration |
| Beam2 | Euler-Bernoulli | 2 | ✅ 6 DOF/node |
| Truss | Axial Bar | 2 | ✅ Axial only |
| Spring/Damper | Discrete | 2 | ✅ Point-to-point |

**Advanced Material Models**:
- ✅ **Von Mises Plasticity** - J2 plasticity with isotropic hardening
- ✅ **Johnson-Cook Plasticity** - Strain rate + thermal softening
- ✅ **Hyperelastic (Neo-Hookean)** - Large deformation rubber-like materials

**Contact Mechanics**:
- ✅ **Penalty Contact** - Node-to-surface with spatial hashing
- ✅ **Coulomb Friction** - Static/dynamic with stick-slip transition
- ✅ **Self-Contact** - Automatic self-contact detection

**Element Erosion & Failure**:
- ✅ Multiple failure criteria (Principal stress/strain, J-C damage, Cockcroft-Latham)
- ✅ Mass redistribution on element deletion
- ✅ Erosion tracking and statistics

**GPU Parallelization**:
- ✅ Kokkos integration - All elements GPU-ready
- ✅ DualView data structures
- ✅ GPU performance: 11 million DOFs/sec

---

## ✅ Wave 3: Implicit Solver (MOSTLY IMPLEMENTED!)

### Phase 3D.1: Tangent Stiffness Assembly ✅ DONE

**Status**: IMPLEMENTED in `fem_static_solver.hpp`

| Task | Description | Status |
|------|-------------|--------|
| Sparse matrix class | CSR format via `SparseMatrix` | ✅ DONE |
| Sparsity pattern | Auto-built from mesh connectivity | ✅ DONE |
| Element assembly | `add_element_matrix()` method | ✅ DONE |
| Hex8 stiffness | Full `stiffness_matrix()` | ✅ DONE |
| Tet4 stiffness | Via `compute_tet4_stiffness()` | ✅ DONE |

**Existing Implementation**:
```cpp
// Already implemented in include/nexussim/solver/implicit_solver.hpp
class SparseMatrix {
    std::vector<Real> values_;
    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptr_;

    void from_coo(...);           // Build from COO format
    void create_pattern(...);      // Pre-compute sparsity
    void add_element_matrix(...);  // FEM assembly
    void multiply(...);            // Matrix-vector product
};
```

### Phase 3D.2: Newton-Raphson Solver ✅ DONE

**Status**: IMPLEMENTED in `implicit_solver.hpp`

| Task | Description | Status |
|------|-------------|--------|
| Residual computation | Callback-based | ✅ DONE |
| Newton iteration | Full implementation | ✅ DONE |
| Line search | Backtracking | ✅ DONE |
| Convergence criteria | Absolute + relative | ✅ DONE |
| Verbose output | Optional | ✅ DONE |

**Algorithm**:
```
1. Initialize: u = u0, iter = 0
2. While not converged and iter < max_iter:
   a. Compute residual: R = F_ext - F_int(u)
   b. Check convergence: ||R|| < tol_R and ||Δu|| < tol_u
   c. Assemble tangent: K = ∂F_int/∂u
   d. Solve: K·Δu = -R
   e. Line search: α = argmin ||R(u + α·Δu)||
   f. Update: u = u + α·Δu
   g. iter++
3. Return u, convergence_status
```

### Phase 3D.3: Linear Solvers ✅ DONE

**Status**: IMPLEMENTED in `implicit_solver.hpp` and `sparse_matrix.hpp`

| Solver | Use Case | Status |
|--------|----------|--------|
| Dense LU | Small problems (<10K DOF) | ✅ `DirectSolver` |
| CG | Large SPD systems | ✅ `CGSolver` |
| GMRES | Non-symmetric | ❌ Not started |
| PETSc | Very large | ❌ Optional |

**Preconditioners** (in `sparse_matrix.hpp`):
| Type | Description | Status |
|------|-------------|--------|
| Jacobi | Diagonal scaling | ✅ `JacobiPreconditioner` |
| SSOR | Symmetric SOR | ✅ `SSORPreconditioner` |
| ILU(0) | Incomplete LU | ❌ Not started |
| AMG | Algebraic multigrid | ❌ Not started |

### Phase 3D.4: Implicit Time Integration ✅ DONE

**Status**: IMPLEMENTED in `implicit_solver.hpp`

**`NewmarkIntegrator` class features**:
- β = 0.25, γ = 0.5 (average acceleration, unconditionally stable)
- Predictor-corrector form
- Rayleigh damping (C = α·M + β·K)
- Integration with Newton-Raphson solver

**`FEMImplicitDynamicSolver` class features** (in `fem_static_solver.hpp`):
- Full Newmark-β FEM dynamic solver
- Mesh-based with automatic stiffness assembly
- Mass matrix (lumped diagonal)
- Energy computation (kinetic + strain)
- BC application via penalty method

### Phase 3D.5: Static Analysis ✅ DONE

**Status**: IMPLEMENTED in `implicit_solver.hpp` and `fem_static_solver.hpp`

**`StaticSolver` class**:
- Load stepping with configurable steps
- Newton-Raphson for nonlinear problems
- Linear solve option (`solve_linear()`)

**`FEMStaticSolver` class**:
- Full mesh-based static solver
- Sparsity pattern from mesh connectivity
- Element stiffness assembly (Hex8, Tet4)
- Dirichlet/Neumann boundary conditions
- Reaction force computation

**Arc-Length Method**: ❌ NOT STARTED (optional, for snap-through buckling)

### Remaining Wave 3 Work

| Task | Priority | Status |
|------|----------|--------|
| Build and run tests | HIGH | ⏳ Needs cmake |
| Validate vs analytical solutions | HIGH | ⏳ Pending |
| Add Hex20/Tet10/Shell stiffness | MEDIUM | ❌ Not started |
| Arc-length continuation | LOW | ❌ Not started |
| PETSc integration | LOW | ❌ Optional |
| GMRES solver | LOW | ❌ Optional |

---

## 📋 Wave 4: Peridynamics Integration (Future)

### Overview

Integrate peridynamics from PeriSys-Haoran for fracture/fragmentation simulation.

**Source Code Location**: `/mnt/d/_working_/FEM-PD/PeriSys-Haoran/code/`

### Phase 4A: Bond-Based Peridynamics

| Task | Reference File | Description |
|------|----------------|-------------|
| Particle data structure | `Global_Para.cuh` | Position, velocity, volume |
| Neighbor list | `JBuildNeighborList.cu` | Horizon-based neighbors |
| Bond force calculation | `JParticle_stress.cu` | Pairwise forces |
| Time integration | `JTime_integral.cu` | Velocity-Verlet |
| Damage model | `JParticle_stress.cu` | Bond breaking |

**Material Models from PeriSys**:
```cpp
enum class MaterialType {
    Elastic = 1,           // Linear elastic
    DruckerPrager = 2,     // Geomaterials
    JohnsonHolmquist2 = 4, // Ceramics/glass
    Rigid = 5,
    JohnsonCook = 7,       // Metals with strain rate
    JohnsonCook_PD = 8,    // PD-specific J-C
    BDPD = 9,              // Bond-based PD
    ElasticBondPD = 10,    // Elastic bonds
    PMMABondPD = 11        // PMMA polymer
};
```

### Phase 4B: State-Based Peridynamics

| Variant | Description | Use Case |
|---------|-------------|----------|
| Ordinary | Force depends on deformation state | General solids |
| Non-ordinary | Correspondence model | Complex materials |
| Dual-horizon | Variable horizon | Multi-scale |

### Phase 4C: PD-FEM Coupling

**Coupling Methods**:

1. **Arlequin Method**
   - Overlapping domain with energy blending
   - Smooth transition from FEM to PD

2. **Bridging Domain**
   - Ghost particles at interface
   - Constraint enforcement

3. **Morphing Coupling**
   - Dynamic FEM-to-PD conversion
   - Based on damage criterion

**Interface Algorithm**:
```
1. Detect interface elements (damage criterion)
2. Create ghost particles in PD domain
3. Apply constraints:
   - Displacement compatibility
   - Force equilibrium
4. Solve coupled system:
   - FEM region: K·u = F_ext - F_coupling
   - PD region: Peridynamic equations + F_coupling
```

### Phase 4D: Crack Propagation

| Feature | Description |
|---------|-------------|
| Bond breaking | Critical stretch/energy criterion |
| Damage tracking | Per-bond damage variable |
| Crack visualization | VTK output with damage field |
| Branching | Natural with PD formulation |

---

## 📊 Priority Matrix

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Sparse matrix assembly | 🔴 High | 🟡 Medium | Pending |
| Newton-Raphson solver | 🔴 High | 🟡 Medium | Pending |
| Linear solvers (CG) | 🔴 High | 🟡 Medium | Pending |
| Newmark-β integration | 🔴 High | 🟢 Low | Pending |
| Static analysis | 🟡 Medium | 🟡 Medium | Pending |
| Bond-based PD | 🔴 High | 🔴 High | Wave 4 |
| PD-FEM coupling | 🔴 High | 🔴 High | Wave 4 |

---

## ✅ Completed Waves Summary

| Wave | Status | Key Features |
|------|--------|--------------|
| Wave 0 | ✅ Complete | Core infrastructure, YAML, VTK |
| Wave 1 | ✅ 75% | Mesh handling, custom format |
| Wave 2 | ✅ Complete | 10 elements, materials, contact, GPU |
| Phase 3A | ✅ Complete | Radioss reader, adaptive timestep |
| Phase 3B | ✅ Complete | Subcycling, consistent mass |
| Phase 3C | ✅ Complete | SPH, FEM-SPH coupling, thermal |

---

## 🎯 Immediate Actions

1. **Start Sparse Matrix Framework**
   - Create `src/solvers/linear/sparse_matrix.hpp`
   - Implement CSR format with Kokkos views

2. **Implement Element Tangent Stiffness**
   - Add `compute_tangent_stiffness()` to element classes
   - Start with Hex8 (simplest 3D element)

3. **Create Newton-Raphson Solver Shell**
   - Basic iteration loop
   - Convergence checking
   - Integration with existing solver interface

---

## 📈 Timeline Estimate

| Milestone | Duration | Deliverable |
|-----------|----------|-------------|
| Sparse matrix + assembly | 1-2 weeks | CSR matrix, element assembly |
| Newton-Raphson | 1-2 weeks | Nonlinear solver working |
| Linear solvers | 1-2 weeks | CG + preconditioner |
| Newmark-β | 1 week | Implicit dynamics |
| Static analysis | 1 week | Static solver |
| **Total Wave 3** | **5-8 weeks** | Full implicit capability |
| PD integration | 4-6 weeks | Bond-based PD |
| PD-FEM coupling | 3-4 weeks | Coupled fracture |
| **Total Wave 4** | **7-10 weeks** | Fracture capability |

---

*Last Updated: 2025-12-28*
*Current Focus: Wave 3 - Implicit Solver*
*Next Focus: Wave 4 - Peridynamics Integration*
