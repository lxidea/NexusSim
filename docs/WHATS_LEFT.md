# What's Left - NexusSim Development Priorities

**Last Updated**: 2026-02-18
**Current Status**: ~92% complete. All core solvers and physics modules done. Remaining work is optional enhancements and infrastructure-blocked items.
**Complete Context**: See `PROJECT_CONTEXT.md` for full project ecosystem

---

## Current State Summary

The project is feature-complete for the core multi-physics mission (FEM + SPH + PD + ALE). All 8 gap closure waves, all PD enhancements, and the implicit solver are done. What remains is either optional, low-priority, or blocked on infrastructure (GPU hardware, MPI cluster).

---

## Completed Since Last Major Update (Dec 2025)

### Implicit Solver — Now ~93% Complete

| Task | Status | Date |
|------|--------|------|
| Element stiffness for Hex20, Tet10 | ✅ Done | Jan 2026 |
| Shell4 proper stiffness (B^T*D*B) | ✅ Done | Jan 2026 |
| Validation test suite (46 checks, 10 tests) | ✅ Done | Feb 2026 |
| J_inv transposition bug fix (all solid elements) | ✅ Done | Feb 2026 |
| hex20_bending_test NaN fix | ✅ Done | Feb 2026 |
| Solver robustness guards (NaN/Inf, diagnostics) | ✅ Done | Feb 2026 |
| Robustness test suite (36 checks, 18 tests) | ✅ Done | Feb 2026 |

### Gap Closure Waves 1-8 — ALL Complete (533 tests)

| Wave | Content | Tests |
|------|---------|-------|
| 1 | 14 material models | 61 |
| 2 | 6 failure models | 52 |
| 3 | Rigid bodies, constraints, rigid walls | 50 |
| 4 | Load curves, load manager, initial conditions | 46 |
| 5 | Tied contact, 5 EOS models | 34 |
| 6 | Restart/checkpoint, enhanced output | 100 |
| 7 | Composite ply stacking, progressive failure | 79 |
| 8 | Sensors, controls, ALE | 111 |

### PD Enhancements — ALL Complete (99 tests)

- Non-ordinary correspondence model (deformation gradient F, 3 constitutive models)
- Enhanced bond models (energy-based, microplastic, viscoelastic, short-range repulsion)
- Dynamic element morphing (FEM-to-PD conversion)
- Mortar + adaptive coupling with damage-driven zone reclassification

---

## Remaining Work

### Priority 1: Implicit Solver Completion (Small)

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Shell4 solver integration | Medium | Medium | Requires 6-DOF support (rotational DOFs) in the solver. Currently only 3-DOF (translational) elements are dispatched. |
| Arc-length method | Low | Medium | Optional — for snap-through/buckling problems |
| PETSc integration | Low | Medium | Optional — for very large problems (>100K DOF) |

**Shell4 6-DOF integration** is the only non-optional item. The Shell4 element already computes correct membrane+bending+shear stiffness via B^T*D*B integration — it just needs the solver to handle 6 DOFs/node instead of 3.

### Priority 2: GPU Performance

| Task | Effort | Blocker |
|------|--------|---------|
| GPU benchmarks | Low | **Requires NVIDIA GPU hardware** (CUDA 12.6 toolkit installed, no device) |
| Memory pool allocator | Medium | None |
| Multi-GPU scaling | High | Requires GPU + MPI |

### Priority 3: Mesh Preprocessing

| Task | Effort | Notes |
|------|--------|-------|
| Auto mesh refinement / coarsening | Medium | h-adaptivity. Mesh quality checks already done (MeshValidator). |

### Priority 4: MPI Parallelization

| Task | Effort | Blocker |
|------|--------|---------|
| Full MPI-parallel solver integration | High | **MPI not installed** on dev machine. Infrastructure (partitioning, ghost exchange) is ready. |
| Scalability benchmarks | Medium | Requires MPI cluster |

---

## What Does NOT Need Work

Everything below is complete and tested:

- All 10 element types (explicit + implicit stiffness for solid elements)
- All 14 standard + 3 PD material models
- All 6 failure models, 5 EOS models
- All contact types (penalty, tied, Hertzian, mortar, self-contact)
- Rigid bodies, constraints (RBE2/RBE3/joints), rigid walls
- Load system (curves, manager, initial conditions)
- Composite ply stacking with progressive failure
- Sensors (5 types, CFC filter), controls (8 action types)
- ALE solver (3 smoothing, 2 advection)
- Restart/checkpoint + enhanced output
- FEM-PD coupling (Arlequin, mortar, direct, morphing, adaptive)
- FEM-SPH coupling
- I/O readers (Radioss, LS-DYNA), VTK writer
- Mesh validation, solver robustness guards
- Kokkos GPU infrastructure

---

## Build & Verify

```bash
# Build
cmake -S . -B build -DNEXUSSIM_ENABLE_MPI=OFF -DNEXUSSIM_BUILD_PYTHON=OFF
cmake --build build -j$(nproc)

# Key validation tests
./build/bin/implicit_validation_test   # 46/46 — implicit solver correctness
./build/bin/fem_robustness_test        # 36/36 — robustness guards
./build/bin/hex20_bending_test         # PASSED — hex20 convergence
```

---

*Last Updated: 2026-02-18*
