# NexusSim Development TODO

**Quick Reference**: Active development priorities
**Complete Context**: See `docs/PROJECT_CONTEXT.md` for full project ecosystem
**Last Updated**: 2026-02-18

---

## Current Status Summary

```
Foundation Waves:
  Wave 0: Foundation           [████████████████████] 100% ✅
  Wave 1: Preprocessing/Mesh   [███████████████░░░░░]  75%
  Wave 2: Explicit Solver      [████████████████████] 100% ✅
  Phase 3A-C: Advanced Physics  [████████████████████] 100% ✅
  Wave 3: Implicit Solver      [███████████████████░]  95%
  Wave 4: Peridynamics         [████████████████████] 100% ✅
  Wave 5: Optimization         [██████████████░░░░░░]  70%
  Wave 6: FEM-PD Coupling      [████████████████████] 100% ✅
  Wave 7: MPI Parallelization  [████████████████░░░░]  80%

Gap Closure Waves (all complete):
  Gap Wave 1: Material Models   [████████████████████] 100% ✅  61 tests
  Gap Wave 2: Failure Models    [████████████████████] 100% ✅  52 tests
  Gap Wave 3: Rigid/Constraints [████████████████████] 100% ✅  50 tests
  Gap Wave 4: Loads System      [████████████████████] 100% ✅  46 tests
  Gap Wave 5: Tied Contact+EOS  [████████████████████] 100% ✅  34 tests
  Gap Wave 6: Checkpoint+Output [████████████████████] 100% ✅ 100 tests
  Gap Wave 7: Composites        [████████████████████] 100% ✅  79 tests
  Gap Wave 8: Sensors/ALE       [████████████████████] 100% ✅ 111 tests

PD Enhancements:
  Correspondence Model          [████████████████████] 100% ✅
  Enhanced Bond Models          [████████████████████] 100% ✅
  Element Morphing              [████████████████████] 100% ✅
  Mortar Coupling               [████████████████████] 100% ✅
  Adaptive Coupling             [████████████████████] 100% ✅  99 tests
```

### Project Metrics

| Metric | Count |
|--------|-------|
| Header files | 124 |
| Test files | 40+ |
| Test assertions | 1,332+ |
| Total LOC | ~87,000 |
| Element types | 10 |
| Material models | 14+ standard + 3 PD-specific |
| Failure models | 6 |
| EOS models | 5 |

---

## ✅ Completed Work

### Foundation (Waves 0-2, Phase 3A-C)

- 10 element types (Hex8, Hex20, Tet4, Tet10, Shell4, Shell3, Wedge6, Beam2, Truss, Spring/Damper)
- Von Mises, Johnson-Cook, Neo-Hookean materials
- Penalty contact with Coulomb friction, node-to-surface, Hertzian, mortar
- Element erosion with multiple failure criteria
- SPH solver with FEM-SPH coupling
- Thermal coupling, subcycling, energy monitoring
- GPU acceleration ready (298M DOFs/sec on OpenMP)

### Implicit Solver (Wave 3)

- Sparse matrix (CSR), CG solver, direct solver
- Jacobi + SSOR preconditioners
- Newton-Raphson with line search
- Newmark-beta integrator
- FEM static solver, FEM implicit dynamic solver
- Element stiffness for Hex8, Hex20, Tet4, Tet10 (all dispatched in solver)
- Shell4 proper membrane+bending+shear stiffness (B^T*D*B integration)
- **Shell4 6-DOF solver integration**: auto-detects shell elements → switches to 6 DOFs/node, local→global stiffness transform (T*K*T^T), rotational penalty for solid-only nodes in mixed meshes
- Fixed J_inv transposition bug in all element shape_derivatives_global()
- Fixed hex20_bending_test NaN: switched from explicit dynamic to static solver, fixed serendipity mesh generation (tensor-product grid created orphan nodes with zero stiffness), added NaN detection
- Robustness guards: NaN/Inf detection in CG solver (RHS, residual, pAp diagnostic), DirectSolver solution scan, element assembly NaN skip, zero diagonal detection, solution NaN scan
- Validation test suite: 46 checks across 10 tests (axial, bending, patch, symmetry, PD)
- Robustness test suite: 36 checks across 18 tests (singular systems, degenerate elements, NaN/Inf propagation)
- Shell4 solver test suite: 24 checks across 7 tests (DOF detection, cantilever bending, membrane tension, symmetry, rotational BCs, patch test, convergence)

### Peridynamics (Wave 4 + Enhancements)

- Bond-based PD (PMB model) with critical stretch failure
- Ordinary state-based PD (dilatation + deviatoric)
- Non-ordinary correspondence model (deformation gradient F, 3 constitutive models)
- Enhanced bond models (energy-based, microplastic, viscoelastic, short-range repulsion)
- Johnson-Cook, Drucker-Prager, Johnson-Holmquist 2 PD materials
- PD contact with spatial hashing and friction
- FEM-PD coupling (Arlequin, mortar, direct force, morphing)
- Adaptive coupling with damage-driven zone reclassification
- Dynamic element morphing (FEM-to-PD conversion)

### Gap Closure (Waves 1-8)

- **Wave 1**: 14 material models (orthotropic, Mooney-Rivlin, Ogden, piecewise-linear, tabulated, foam, crushable foam, honeycomb, viscoelastic, Cowper-Symonds, Zhao, elastic-plastic-fail, rigid, null)
- **Wave 2**: 6 failure models (Hashin, Tsai-Wu, Chang-Chang, GTN, GISSMO, tabulated)
- **Wave 3**: Rigid bodies, constraints (RBE2/RBE3/joints), rigid walls (planar/cylindrical/spherical/moving)
- **Wave 4**: Load curves, load manager, initial conditions
- **Wave 5**: Tied contact (with failure), EOS (ideal gas, Gruneisen, JWL, polynomial, tabulated)
- **Wave 6**: Restart/checkpoint (basic + extended), enhanced output (time history, result database, cross-section force, interface force, part energy)
- **Wave 7**: Composite ply stacking (thermal residual stress, interlaminar shear, progressive failure/FPF/strength envelope)
- **Wave 8**: Sensors (5 types, CFC filter), controls (8 action types), ALE (3 smoothing, 2 advection)

### I/O and Readers

- Radioss legacy format reader
- LS-DYNA k-file reader (~30 keywords across 8 groups, 171 tests)
- VTK animation writer
- Checkpoint files (basic + extended)

### Performance

- Benchmarking infrastructure (Timer, ScopedTimer, Profiler, MemoryStats)
- Kokkos multi-backend support (OpenMP active, CUDA/HIP code-ready)

---

## Remaining Work

### Priority 1: Implicit Solver Completion

- [x] Element stiffness matrices for Hex20, Tet10 (dispatched in solver)
- [x] Shell4 proper stiffness (membrane + bending + shear B^T*D*B)
- [x] Validation against analytical solutions (axial, cantilever, patch tests)
- [x] Fix J_inv transposition bug in all solid elements (hex8, hex20, tet4, tet10, wedge6)
- [x] Shell4 solver integration (6-DOF auto-detection, local→global transform, mixed mesh support)
- [ ] Arc-length method for snap-through buckling (optional)
- [ ] PETSc integration for very large problems (optional)

- [x] Robustness guards (NaN/Inf detection, degenerate element handling, solver diagnostics)

### Priority 2: GPU Performance

- [ ] GPU benchmarks (requires NVIDIA GPU hardware)
- [ ] Memory pool allocator to reduce allocation overhead
- [ ] Multi-GPU scaling (MPI + Kokkos distributed GPU)

### Priority 3: Mesh Preprocessing (Wave 1 Remaining)

- [x] Mesh quality checks and element Jacobian validation (MeshValidator + assembly guards)
- [ ] Automatic mesh refinement / coarsening

### Priority 4: MPI (Wave 7 Remaining)

- [ ] Full MPI-parallel solver integration (infrastructure is ready)
- [ ] Scalability benchmarks across multiple nodes

---

## Architecture Overview

### Header Organization (124 files)

| Subdirectory | Headers | Key Contents |
|-------------|---------|-------------|
| `physics/` | 42 | Materials (14), failure (9), composites (5), EOS, erosion, thermal |
| `fem/` | 18 | Contact, constraints, rigid bodies, loads, sensors, controls |
| `peridynamics/` | 15 | PD types, particle, neighbor, force, solver, coupling, correspondence |
| `io/` | 14 | Readers (Radioss, LS-DYNA), VTK writer, checkpoint, output |
| `discretization/` | 11 | Hex8/20, Tet4/10, Shell3/4, Wedge6, Beam2, Truss, Spring |
| `core/` | 7 | Types, logger, memory, GPU, MPI, exceptions |
| `solver/` | 4 | Implicit solver, sparse matrix, GPU sparse, FEM static |
| `data/` | 4 | Mesh, state, field |
| `sph/` | 4 | SPH solver, kernel, neighbor search, FEM-SPH coupling |
| `coupling/` | 3 | Coupling operators, field registry |
| `ale/` | 1 | ALE solver |
| `utils/` | 1 | Performance timer |

### Test Files

| Test | Assertions | Area |
|------|-----------|------|
| `lsdyna_reader_ext_test.cpp` | 172 | LS-DYNA reader extensions |
| `sensor_ale_test.cpp` | 104 | Sensors, controls, ALE |
| `restart_output_test.cpp` | 110 | Checkpoint + output |
| `enhanced_output_test.cpp` | 97 | Extended output modules |
| `pd_enhanced_test.cpp` | 99 | PD correspondence, bonds, morphing, coupling |
| `composite_layup_test.cpp` | 83 | Composite layup system |
| `composite_progressive_test.cpp` | 78 | Progressive failure |
| `realistic_crash_test.cpp` | 63 | Multi-system integration |
| `material_models_test.cpp` | 62 | 14 material models |
| `hertzian_mortar_test.cpp` | 53 | Hertzian + mortar contact |
| `failure_models_test.cpp` | 53 | 6 failure models |
| `rigid_body_test.cpp` | 51 | Rigid bodies + constraints |
| `loads_system_test.cpp` | 47 | Load curves + initial conditions |
| `pd_fem_coupling_test.cpp` | 39 | FEM-PD coupling |
| `tied_contact_eos_test.cpp` | 35 | Tied contact + EOS |
| `fem_pd_integration_test.cpp` | 29 | FEM-PD integration |
| `implicit_validation_test.cpp` | 46 | Implicit solver multi-element validation |
| `fem_robustness_test.cpp` | 36 | Solver robustness guards (NaN, singular, degenerate) |
| `shell4_solver_test.cpp` | 24 | Shell4 6-DOF solver integration (bending, membrane, BCs, convergence) |
| `mpi_partition_test.cpp` | 23 | MPI partitioning |

---

*Last Updated: 2026-02-18*
