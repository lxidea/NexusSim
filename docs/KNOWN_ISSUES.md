# Known Issues - NexusSim

This document tracks known bugs and limitations in the current release.

**Last Updated**: 2025-11-07

---

## Active Issues

### 1. Hex20 Element NaN Instability (HIGH PRIORITY)

**Status**: 🔴 Active Bug
**Severity**: High
**Affects**: Hex20 (20-node quadratic hexahedral element)

**Description**:
Hex20 element produces NaN values during time integration at approximately step 319 in dynamic simulations.

**Symptoms**:
- Simulation runs normally for ~319 timesteps
- NaN suddenly appears in displacement field
- Occurs even with:
  - Single element (no orphan nodes)
  - Correct mass matrix (verified via tests)
  - Rayleigh damping added
  - Zero-mass DOF constraints in place

**Root Cause Analysis** (Updated 2025-11-07):
- Mass matrix calculation: ✅ Verified correct (3D consistent mass = 3 × ρV)
- Shape functions: ✅ Validated (partition of unity holds)
- Volume calculation: ✅ Accurate (<1e-14% error)
- Jacobian computation: ✅ Positive definite
- **IDENTIFIED**: Numerical instability in force calculation causing exponential growth
  - Displacements grow: 1e-8 → 1e+135 → 1e+170 → 1e+205 → NaN
  - Suggests incorrect stress/stiffness sign or hourglass mode
  - Element "pushes" instead of resisting deformation

**Evidence**:
```
Test Results (hex20_single_element_test):
✓ Shape functions: PASS
✓ Mass matrix: PASS (3000 kg for 1m³ @ 1000 kg/m³ - correct for 3 DOFs)
✓ Volume: PASS
✓ Jacobian: PASS

Dynamic Simulation (hex20_single_element_bending):
✗ NaN at step 319
```

**Workaround**:
Use Tet10 elements for quadratic accuracy needs until Hex20 is fixed.

**Next Steps**:
1. Debug element force calculation in `src/discretization/fem/solid/hex20.cpp`
2. Verify B-matrix computation at all integration points
3. Check for numerical instabilities in strain-displacement calculations
4. Compare against working Hex8 element implementation

**Related Files**:
- `src/discretization/fem/solid/hex20.cpp` (lines 400-700)
- `examples/hex20_single_element_bending.cpp`
- `examples/hex20_single_element_test.cpp`

---

## Resolved Issues

### ✅ Hex20 Mass Matrix Test Failure (RESOLVED 2025-11-07)

**Status**: ✅ Fixed
**Resolution**: Test was incorrect, not implementation

**Issue**: Test expected mass matrix sum = ρV, got 3×ρV
**Fix**: Consistent mass matrix in 3D correctly sums to 3×ρV (one for each DOF)
**Commit**: Test updated to check `sum(M) = 3 × ρ × V`

### ✅ Mesh Generation Creating Orphan Nodes (RESOLVED 2025-11-07)

**Status**: ✅ Workaround implemented
**Resolution**: Zero-mass DOF detection and constraint system

**Issue**: Structured mesh generation created unused nodes with zero mass
**Fix**: FEM solver now detects zero-mass DOFs and constrains them
**Files Modified**:
- `src/fem/fem_solver.cpp` (lines 349-401, 204-219)
- `include/nexussim/fem/fem_solver.hpp` (line 268)

---

## Limitations (By Design)

### 1. Element Library

**Production Ready** (6/7 = 85%):
- ✅ Hex8 - 8-node hexahedral (linear)
- ✅ Tet4 - 4-node tetrahedral (linear)
- ✅ Tet10 - 10-node tetrahedral (quadratic)
- ✅ Shell4 - 4-node shell
- ✅ Wedge6 - 6-node prism
- ✅ Beam2 - 2-node beam

**Partially Working** (1/7):
- ⚠️ Hex20 - 20-node hexahedral (quadratic) - Has NaN bug

### 2. Material Models

**Currently Available**:
- ✅ Linear elastic (isotropic)

**Planned**:
- ⏳ Johnson-Cook plasticity
- ⏳ Neo-Hookean hyperelasticity
- ⏳ Mooney-Rivlin
- ⏳ Rate-dependent materials

### 3. Solver Capabilities

**Available**:
- ✅ Explicit time integration (central difference)
- ✅ GPU acceleration (CUDA, OpenMP, Serial via Kokkos)
- ✅ Basic boundary conditions (displacement, force)

**Not Yet Implemented**:
- ⏳ Implicit solver (Newmark-β, HHT-α)
- ⏳ Contact mechanics (penalty, Lagrange multiplier)
- ⏳ Multi-physics coupling
- ⏳ Adaptive timestepping

### 4. I/O Formats

**Available**:
- ✅ VTK output (visualization)
- ✅ YAML configuration
- ✅ Simple mesh format

**Planned**:
- ⏳ Radioss input format
- ⏳ LS-DYNA k-file
- ⏳ Abaqus input
- ⏳ HDF5 output

---

## Performance Notes

### GPU Performance (CUDA Backend)

**Confirmed Working** (2025-11-07):
- Default execution space: `Kokkos::Cuda`
- Peak throughput: 12.4 million DOFs/sec (27000 elements)
- All element kernels GPU-accelerated
- Atomic assembly for thread-safe force accumulation

**Benchmark Results**:
| Problem Size | Elements | DOFs | Time/Step | Throughput |
|--------------|----------|------|-----------|------------|
| Small | 25 | 216 | 0.29 ms | 751k DOFs/sec |
| Medium | 100 | 726 | 0.48 ms | 1.5M DOFs/sec |
| Large | 800 | 3969 | 0.47 ms | 8.5M DOFs/sec |
| Very Large | 2700 | 11532 | 0.93 ms | 12.4M DOFs/sec |

---

## Reporting New Issues

Please report issues at: https://github.com/nexussim/nexussim/issues

Include:
1. Element type and mesh size
2. Material properties
3. Boundary conditions
4. Steps to reproduce
5. Log output (set `NXS_LOG_LEVEL=DEBUG`)

---

*Document Version: 1.0*
*Last Updated: 2025-11-07*
