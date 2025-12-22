# Getting Started with Next Development Phase

## Quick Start for Contributors

This guide helps you jump into the next phase of NexusSim development.

---

## Where We Are (Updated 2025-11-07)

**Current Status**: Wave 0 Complete + Wave 1 85% + Wave 2 70%

**Working Components**:
- ✅ Hex8 element (production-ready)
- ✅ Tet4 element (production-ready)
- ✅ Tet10 element (production-ready)
- ✅ Shell4 element (production-ready)
- ✅ Wedge6 element (production-ready)
- ✅ Beam2 element (production-ready)
- ✅ Explicit central difference solver (GPU-parallelized)
- ✅ GPU kernels (80% implemented - Kokkos parallel loops)
- ✅ Basic I/O (mesh, VTK output)
- ✅ YAML configuration

**What's Close**:
- ⚠️ Hex20 element (90% - mesh generation bug, element code is correct)
- ⚠️ GPU backend verification (code ready, need to confirm CUDA/HIP enabled)

**What's Missing** (Lower Priority):
- ❌ Production format readers (Radioss, LS-DYNA)
- ❌ Contact mechanics (code exists, needs testing)
- ❌ Advanced materials (only elastic implemented)

---

## Top 3 Priority Tasks (Updated 2025-11-07)

### 🎯 Priority 1: Fix Hex20 Mesh Generation Bug

**Why**: Hex20 element is fully implemented, but test uses faulty mesh generator

**Effort**: 2-4 hours (just fix mesh generation, element code is correct!)

**Status**: ✅ Element implementation complete, ⚠️ Mesh generation creates orphan nodes

**Root Cause**:
- `hex20_bending_test.cpp` creates structured grid with unused nodes
- For 2×1×1 elements: creates 45 nodes but only ~31 are connected
- Unused nodes get zero mass → NaN in time integration

**Files to Modify**:
```
examples/hex20_bending_test.cpp  (Fix mesh generation)
```

**Implementation Options**:
- Option A: Rewrite mesh generation to only create connected nodes (tedious but certain)
- Option B: Integrate Gmsh or similar mesh generator
- Option C: Create hex20 mesh utilities library

**Safety Fix Already Implemented**:
- Zero-mass DOFs are detected and constrained (lines 349-401 in fem_solver.cpp)
- Prevents immediate crashes but doesn't fix underlying mesh issue

**Expected Outcome**:
- Hex20 test runs to completion without NaN
- All 7 element types production-ready

---

### 🎯 Priority 2: Verify GPU Backend Configuration

**Why**: Code is GPU-ready, just need to verify backend compilation

**Effort**: 30 minutes - 1 hour

**Status**: ✅ GPU kernels 80% implemented, ⚠️ Backend verification needed

**What's Already Done**:
- ✅ All state vectors use `Kokkos::DualView`
- ✅ Time integration parallelized (line 173)
- ✅ All element types have GPU kernels (lines 468, 645, 817, 1092, 1321)
- ✅ Atomic assembly for thread-safety
- ✅ GPU memory management (`sync_device()`, `modify_device()`)

**What Needs Verification**:
```bash
# Check CMake configuration
cd build
cmake .. -LAH | grep -i kokkos

# Look for:
# - Kokkos_ENABLE_CUDA=ON (or HIP, or OpenMP)
# - Kokkos execution space

# Run simple test to confirm GPU execution
./bin/hex8_element_test  # Should use GPU if enabled
```

**Implementation Checklist**:
- [ ] Check CMakeLists.txt for Kokkos backend flags
- [ ] Verify `find_package(Kokkos)` configuration
- [ ] Run test with GPU backend enabled
- [ ] Benchmark CPU vs GPU performance
- [ ] Add GPU utilization monitoring

**Expected Outcome**:
- Confirm actual GPU execution (not just CPU OpenMP)
- Measure 10-100x speedup on large models
- Scalable to multi-GPU (future work)

---

### 🎯 Priority 3: Implement Radioss Format Reader

**Why**: Legacy compatibility, enables migration from OpenRadioss

**Effort**: 1-2 weeks

**Files to Create**:
```
src/io/readers/radioss/radioss_reader.cpp  (NEW)
include/nexussim/io/radioss_reader.hpp  (NEW)
tests/io/test_radioss_reader.cpp  (NEW)
```

**Radioss Input Format Structure**:
```
/BEGIN
/TITLE
Example simulation

/NODE
   NodeID       X          Y          Z
        1    0.0        0.0        0.0
        2    1.0        0.0        0.0
   ...

/ELEM/SOLID
   ElemID  NodeIDs...
        1   1  2  3  4  5  6  7  8
   ...

/MAT/ELASTIC
   MatID  Rho  E  Nu
       1  7850  210e9  0.3
   ...

/BCS/FIX
   NodeID  Tx  Ty  Tz
        1   1   1   1
   ...

/END
```

**Implementation Checklist**:
- [ ] Tokenizer/lexer for Radioss format
- [ ] Parse `/NODE` block → Mesh nodes
- [ ] Parse `/ELEM/*` blocks → Elements
- [ ] Parse `/MAT/*` blocks → Materials
- [ ] Parse `/BCS/*` blocks → Boundary conditions
- [ ] Parse `/LOAD/*` blocks → Applied loads
- [ ] Create `Mesh` object from parsed data
- [ ] Validation against OpenRadioss test decks

**Expected Outcome**:
- Load OpenRadioss input files directly
- 100+ legacy test cases available

---

## How to Choose What to Work On

### If You Have FEM Background
→ **Implement Hex20 or Shell4 element**

### If You Have GPU/HPC Background
→ **Activate GPU kernels with Kokkos**

### If You Have Parsing/I/O Experience
→ **Implement Radioss format reader**

### If You Have Contact Mechanics Knowledge
→ **Implement penalty contact** (see roadmap doc)

---

## Development Setup

### 1. Build Current Code

```bash
cd /mnt/d/_working_/FEM-PD/claude-radioss
mkdir build && cd build
cmake ..
make -j4
```

### 2. Run Tests

```bash
./bin/bending_test          # Bending validation
./bin/patch_test            # FEM patch test
./bin/fem_solver_test       # Basic solver test
```

### 3. Create Feature Branch

```bash
git checkout -b feature/hex20-element
# ... implement feature ...
git commit -m "Implement Hex20 quadratic element"
```

---

## Testing Your Implementation

### Unit Test Template

```cpp
// tests/unit/test_hex20.cpp
#include <catch2/catch.hpp>
#include <nexussim/discretization/hex20.hpp>

TEST_CASE("Hex20 shape functions at center", "[hex20]") {
    Hex20Element elem;
    Real xi[3] = {0.0, 0.0, 0.0};  // Element center
    Real N[20];

    elem.shape_functions(xi, N);

    // Sum of shape functions = 1
    Real sum = 0.0;
    for (int i = 0; i < 20; ++i) sum += N[i];
    REQUIRE(sum == Approx(1.0));
}
```

### Integration Test Template

```cpp
// tests/integration/test_hex20_solver.cpp
TEST_CASE("Hex20 bending test", "[integration]") {
    // Create mesh with Hex20 elements
    auto mesh = create_cantilever_mesh_hex20();

    // Setup solver
    FEMSolver solver("Test");
    solver.add_element_group("beam", ElementType::Hex20, ...);
    solver.initialize(mesh, state);

    // Run simulation
    for (int step = 0; step < 100; ++step) {
        solver.step(dt);
    }

    // Validate deflection
    Real tip_deflection = solver.displacement()[tip_node * 3 + 2];
    Real analytical = compute_analytical_deflection();

    REQUIRE(std::abs(tip_deflection - analytical) / analytical < 0.05);
}
```

---

## Documentation Requirements

### For Each New Feature

1. **API Documentation** (Doxygen)
```cpp
/**
 * @brief 20-node quadratic hexahedral element
 *
 * Features:
 * - Quadratic shape functions (no locking)
 * - 2×2×2 Gauss integration (default)
 * - 3×3×3 integration available
 *
 * Node numbering:
 *   Corner nodes: 0-7 (same as Hex8)
 *   Mid-edge nodes: 8-19
 *
 * @see Hex8Element for linear variant
 */
class Hex20Element : public Element { ... };
```

2. **User Guide Section**
```markdown
# Hex20 Element

## When to Use
- Bending-dominated problems
- Curved geometries
- High accuracy requirements

## Integration Schemes
- Default: 2×2×2 (8 points)
- High accuracy: 3×3×3 (27 points)

## Example
...
```

3. **Update Roadmap**
```markdown
| Hex20 | ✅ Complete | Quadratic solid element | - |
```

---

## Code Style Guidelines

### C++20 Modern Practices

```cpp
// Use auto for type deduction
auto mesh = std::make_shared<Mesh>(num_nodes);

// Use range-based for loops
for (const auto& elem : element_groups) { ... }

// Use std::span for array views
void process(std::span<const Real> coords) { ... }

// Use [[nodiscard]] for important returns
[[nodiscard]] Real compute_stable_dt() const;

// Use constexpr for compile-time constants
static constexpr int NUM_NODES = 20;
```

### Kokkos GPU Compatibility

```cpp
// Mark GPU functions
KOKKOS_INLINE_FUNCTION
void shape_functions(const Real xi[3], Real* N) const {
    // No std::vector, no dynamic allocation
    // Use stack arrays or passed-in memory
}

// Use Kokkos Views for data
Kokkos::View<Real*> displacement_("disp", ndof);

// Use parallel loops
Kokkos::parallel_for("ElementLoop", num_elems,
    KOKKOS_LAMBDA(const int e) {
        // Element computation
    });
```

---

## Getting Help

### Documentation
- **Architecture**: `docs/Framework_Architecture_Current_State.md`
- **Integration**: `docs/Element_Integration_Strategies.md`
- **Roadmap**: `docs/Development_Roadmap_Status.md`

### Code Examples
- **Hex8 element**: `src/discretization/fem/solid/hex8.cpp`
- **FEM solver**: `src/fem/fem_solver.cpp`
- **Example programs**: `examples/*.cpp`

### Reference Materials
- **FEM Theory**: Hughes "The Finite Element Method"
- **Kokkos**: https://kokkos.org/kokkos-core-wiki/
- **OpenRadioss**: Reference implementation

---

## Quick Wins for New Contributors

### Easy (1-2 days)
- [ ] Add mesh quality metrics (aspect ratio, Jacobian)
- [ ] Implement VTK timestep series (PVD files)
- [ ] Add more material models (hyperelastic, plasticity)
- [ ] Improve logging (more detailed diagnostics)

### Medium (3-5 days)
- [ ] Implement Tet4 element
- [ ] Add HDF5 output support
- [ ] Implement adaptive timestepping
- [ ] Add consistent mass matrix option

### Hard (1-2 weeks)
- [ ] Implement Hex20 element
- [ ] Activate GPU kernels
- [ ] Implement contact mechanics
- [ ] Add Radioss format reader

---

## Success Metrics

### How to Know You're Done

**Element Implementation**:
- [ ] Patch test passes (<1% error)
- [ ] Bending test passes (<5% error)
- [ ] Convergence study shows expected rate
- [ ] GPU kernel works (if applicable)

**GPU Activation**:
- [ ] Achieves 10x+ speedup vs CPU
- [ ] Correctness verified (matches CPU results)
- [ ] Scalability demonstrated (multi-GPU)

**Format Reader**:
- [ ] Loads 10+ OpenRadioss test decks
- [ ] Results match OpenRadioss output
- [ ] Error handling for malformed input

---

## Next Review Checkpoint

**Date**: 2025-11-06 (1 week)

**Agenda**:
- Progress on priority tasks
- Blockers and issues
- Roadmap adjustments
- New priority assignments

---

*Good luck and happy coding!* 🚀

*Questions? Check the docs/ directory or ask the team.*
