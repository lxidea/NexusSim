# Element Integration Strategies

## Overview

The NexusSim FEM framework uses a **flexible, element-specific integration approach**. Each element type defines its own optimal Gauss quadrature scheme through the `gauss_quadrature()` method, allowing different elements to use different integration strategies based on their formulation and intended use cases.

## Architecture

### Design Principle

**Integration is an element property, not a solver property.**

- The `FEMSolver` queries each element for its integration points via `elem->gauss_quadrature()`
- Each element returns the appropriate number of points and weights
- The solver loops over these points to compute internal forces
- Different element types can use different integration schemes

### Implementation

**File**: `src/fem/fem_solver.cpp:270-329`

```cpp
// Get element's integration points and weights
const int num_gp = props.num_gauss_points;
std::vector<Real> gp_coords(num_gp * 3);
std::vector<Real> gp_weights(num_gp);
elem->gauss_quadrature(gp_coords.data(), gp_weights.data());

// Loop over integration points
for (int gp_idx = 0; gp_idx < num_gp; ++gp_idx) {
    // Use element-specified integration point
    Real xi[3] = {...};
    Real weight = gp_weights[gp_idx];
    // ... compute forces at this point
}
```

## Element-Specific Integration Schemes

### Hex8 (8-Node Hexahedron)

**Current Strategy**: 1-point reduced integration

**File**: `src/discretization/fem/solid/hex8.cpp:123-126`

```cpp
void Hex8Element::gauss_quadrature(Real* points, Real* weights) const {
    // Default to 1-point integration for explicit dynamics
    compute_gauss_points_1pt(points, weights);
}
```

**Rationale**:
- ✅ **Avoids volumetric locking** in bending and nearly-incompressible materials
- ✅ **Faster computation** for explicit dynamics (1 vs 8 evaluations)
- ⚠️ **Hourglass modes** require stabilization (currently disabled)
- ✅ **Good for**: tension, compression, moderate bending, impact
- ❌ **Poor for**: pure bending of slender structures

**Alternative Available**: 8-point full integration (`compute_gauss_points_8pt()`)
- Provides locking but avoids hourglass modes
- Can be activated by changing `gauss_quadrature()` implementation

### Hex20 (20-Node Quadratic Hexahedron)

**Recommended Strategy**: 2×2×2 or 3×3×3 full integration

**Status**: ⚠️ Not yet implemented (placeholder exists)

**Rationale**:
- Quadratic elements need more integration points for accuracy
- Full integration appropriate (no locking with quadratic shape functions)
- 2×2×2 = 8 points (sufficient for most cases)
- 3×3×3 = 27 points (high accuracy, expensive)

### Shell Elements

**Recommended Strategy**:
- In-plane: 2×2 Gauss (for 4-node shells)
- Through-thickness: Simpson's rule or 2-5 points

**Status**: ⚠️ Not yet implemented

**Rationale**:
- In-plane integration captures membrane and bending behavior
- Through-thickness integration for composite layers

### Beam Elements

**Recommended Strategy**: 2-3 point Gauss along length

**Status**: ⚠️ Not yet implemented

**Rationale**:
- 2-point sufficient for linear beams
- 3-point for nonlinear material behavior

## Integration Point Selection Guidelines

### General Rules

| Element Order | Min Points Required | Recommended | Notes |
|---------------|-------------------|-------------|-------|
| Linear (Hex8, Tet4) | 1 | 1 (reduced) or 2×2×2 (full) | Trade-off: locking vs hourglass |
| Quadratic (Hex20, Tet10) | 2×2×2 | 2×2×2 or 3×3×3 | Full integration preferred |
| Shell (linear) | 2×2 in-plane | 2×2 × (2-5 thick) | Depends on laminate |
| Beam (linear) | 2 | 2-3 | 3 for plasticity |

### Accuracy vs Performance

**1-Point Integration**:
- Pros: Fastest, avoids locking, good for explicit dynamics
- Cons: Hourglass modes, requires stabilization
- Best for: High-speed impact, large deformations with coarse mesh

**Full Integration** (2×2×2 for hex):
- Pros: No hourglass, stable, accurate for smooth solutions
- Cons: Volumetric locking (linear elements), 8× slower than 1-point
- Best for: Static analysis, fine meshes, quadratic elements

**Selective Reduced Integration (SRI)**:
- Pros: Best of both worlds (full for volumetric, reduced for deviatoric)
- Cons: More complex implementation
- Best for: Production-quality simulations
- Status: ⚠️ Future enhancement

## Switching Integration Schemes

### At Element Level (Recommended)

Modify the element's `gauss_quadrature()` method:

```cpp
// In hex8.cpp
void Hex8Element::gauss_quadrature(Real* points, Real* weights) const {
    // Option 1: 1-point (current default)
    compute_gauss_points_1pt(points, weights);

    // Option 2: 8-point full integration (uncomment to use)
    // compute_gauss_points_8pt(points, weights);
}
```

Also update `properties().num_gauss_points` to match:

```cpp
Properties properties() const override {
    return Properties{
        ...
        1,  // num_gauss_points: change to 8 for full integration
        ...
    };
}
```

### Future: Runtime Configuration

Potential enhancement - allow integration scheme as a parameter:

```cpp
// Future API (not yet implemented)
auto hex8_reduced = std::make_shared<Hex8Element>(IntegrationScheme::Reduced);
auto hex8_full = std::make_shared<Hex8Element>(IntegrationScheme::Full);
```

## Benchmarks and Validation

### Hex8 with 1-Point Integration

**Test**: Cantilever beam bending (`examples/bending_test.cpp`)

| Mesh | Elements | Result | Error | Status |
|------|----------|--------|-------|--------|
| 4×1×2 | 8 | -64.5 μm | 13% | ✅ Acceptable |
| 8×2×2 | 32 | Unstable | >2000% | ❌ Hourglass |
| 12×2×3 | 72 | Unstable | >1600% | ❌ Hourglass |

**Conclusion**: Hex8 1-point acceptable for **coarse meshes** only. Fine meshes require hourglass control or different element type.

### Hex8 with 8-Point Integration (Historical)

**Test**: Same cantilever beam

| Mesh | Elements | Result | Error | Status |
|------|----------|--------|-------|--------|
| 4×1×2 | 8 | -15.8 μm | 72% | ❌ Locking |
| 8×2×2 | 32 | Unstable | >1800% | ❌ Locking |

**Conclusion**: Full integration causes **severe volumetric locking** for bending. Not recommended for linear hex elements.

## Hourglass Control

### Status

**Implementation**: ✅ Available (`Hex8Element::hourglass_forces()`)
**Activation**: ❌ Currently disabled in `FEMSolver`

**Location**: `src/discretization/fem/solid/hex8.cpp:552-623`

### Future Activation

To enable hourglass control:

1. Uncomment hourglass code in `fem_solver.cpp:326-338`
2. Tune `hg_stiffness` parameter (currently 0.0005 × shear modulus)
3. Validate with hourglass benchmark tests

### Limitations

Current implementation shows:
- ✅ Works for compression-dominated problems
- ❌ Insufficient for pure bending of slender beams
- ⚠️ Too weak → hourglass modes persist
- ⚠️ Too strong → artificial stiffening

**Recommended approach**: Use Hex20 for bending instead of hourglass control on Hex8.

## Recommendations by Problem Type

### Impact/Crash (Explicit Dynamics)

- **Element**: Hex8
- **Integration**: 1-point reduced
- **Hourglass**: Light stabilization (0.05-0.1% of bulk modulus)
- **Mesh**: Coarse acceptable (goal: fast timestep)

### Structural Bending (Static/Implicit)

- **Element**: Hex20 or shells
- **Integration**: Full (2×2×2 for Hex20)
- **Hourglass**: Not needed
- **Mesh**: Moderate refinement

### Forming/Large Deformation

- **Element**: Hex8 with selective reduced integration (future)
- **Integration**: Mixed (full volumetric, reduced deviatoric)
- **Hourglass**: As needed
- **Mesh**: Adaptive refinement

### Nearly Incompressible (Rubber, Plasticity)

- **Element**: Hex8 reduced or mixed formulation
- **Integration**: 1-point or B-bar method
- **Hourglass**: Essential
- **Mesh**: Fine with aspect ratio control

## References

1. **Reduced Integration**: Hughes, T.J.R. "The Finite Element Method" (2000)
2. **Hourglass Control**: Flanagan & Belytschko, IJNME (1981)
3. **Selective Reduced Integration**: Zienkiewicz & Taylor, Vol 2 (2000)
4. **B-bar Method**: Hughes, CMAME (1980)

## Future Enhancements

### Roadmap

- [ ] **Implement Hex20** with full 2×2×2 or 3×3×3 integration
- [ ] **Selective Reduced Integration (SRI)** for Hex8
- [ ] **Runtime integration scheme selection** per element
- [ ] **Adaptive integration** based on error indicators
- [ ] **Hourglass control tuning** and benchmarking
- [ ] **Mixed/enhanced strain formulations** (EAS, B-bar)

### Contributing

When adding new element types, implement:

1. `gauss_quadrature()` method with appropriate scheme
2. `compute_gauss_points_Xpt()` helper functions for different schemes
3. Update `properties().num_gauss_points` accordingly
4. Document the choice in this file
5. Add benchmark tests validating the integration scheme

---

*Last Updated: 2025-10-30*
*Maintainer: NexusSim Development Team*
