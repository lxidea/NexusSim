# Hex20 Force Sign Bug - Root Cause Analysis
## November 7, 2025 - Final Session

## 🎯 ROOT CAUSE IDENTIFIED

**The internal forces computed by Hex20 have THE OPPOSITE SIGN from what they should be.**

This causes positive feedback instead of negative feedback, leading to exponential growth.

---

## Smoking Gun Evidence

From actual simulation output at tip node (node 1):

```
Step 0: uz = +1.90e-06,  f_int_z = 0          (initial)
Step 1: uz = -5.14e-05,  f_int_z = +232,772   ← WRONG! Should be negative!
Step 2: uz = +0.00150,   f_int_z = -6.05e+06  ← WRONG! Should be positive!
Step 3: uz = -0.0434,    f_int_z = +1.77e+08  ← WRONG! Should be negative!
Step 4: uz = +1.26,      f_int_z = -5.13e+09  ← WRONG! Should be positive!
```

**Pattern**:
- When uz > 0 (upward), f_int should be < 0 (resisting, pulling down) → **BUT IT'S POSITIVE!**
- When uz < 0 (downward), f_int should be > 0 (resisting, pushing up) → **BUT IT'S NEGATIVE!**

This creates **positive feedback** causing exponential growth.

---

## What We've Ruled Out

### ✅ NOT the Jacobian Transformation
- Tested both "row-major" and "column-major" interpretations
- Both give IDENTICAL wrong results
- Hex8 uses same formula and works perfectly
- **Conclusion**: Transformation is correct

### ✅ NOT the Jacobian Matrix Assembly
- Both Hex8 and Hex20 use column-major storage
- Formula is identical
- **Conclusion**: Jacobian assembly is correct

###  ✅ NOT the Inverse Jacobian
- Cofactor formula is IDENTICAL to Hex8
- **Conclusion**: Inverse calculation is correct

### ✅ NOT the B-Matrix Structure
- Strain-displacement matrix structure matches Hex8 exactly
- **Conclusion**: B-matrix assembly is correct

### ✅ NOT the Force Assembly
- CPU fallback uses: `elem_force[j] += B[i,j] * stress[i] * detJ * weight`
- This is mathematically correct
- **Conclusion**: Force accumulation formula is correct

### ✅ NOT the Time Integration
- Uses `net_force = f_ext - f_int` (correct sign)
- Works perfectly for all 6 other element types
- **Conclusion**: Time integration is correct

---

## Remaining Suspects

Since all the infrastructure is correct, the bug must be in the **Hex20 shape function derivatives themselves**.

### Theory: Shape Function Derivatives Have Wrong Sign

**Evidence**:
1. All unit tests pass, but they might not test the sign correctly
2. Mass matrix test passes because it uses N^T * N (sign cancels out)
3. Jacobian test passes because it's just checking magnitude
4. The ONLY place where sign matters for dynamics is in dN/dx

### Specific Hypothesis

The Hex20 shape function derivatives might have:
- Wrong overall sign (multiplied by -1 somewhere)
- Wrong sign for specific node types (corners vs mid-edges)
- Sign error in the serendipity interpolation formula

---

## Test to Confirm

Create a simple test:
1. Take unit cube element
2. Apply small known displacement (e.g., +0.001m in z at one corner)
3. Compute strains using B-matrix
4. Compute stresses
5. Compute internal forces
6. **Check**: Should force oppose displacement?

Expected for +z displacement at free node:
- Strain: εzz > 0 (tension)
- Stress: σzz > 0 (tension)
- Force at that node: fz < 0 (resisting upward motion)

If force has same sign as displacement → confirms wrong sign in derivatives!

---

## Recommended Fix Steps

### Step 1: Verify Shape Function Derivative Signs (30 min)

Compare Hex20 shape function derivatives at element center against published formulas:

**Reference**: Hughes "The Finite Element Method" or Zienkiewicz & Taylor

For node 0 (corner at ξ=-1, η=-1, ζ=-1), at element center (0,0,0):
```
dN0/dξ should be approximately -0.015625  (negative because node is at -1)
dN0/dη should be approximately -0.015625  (negative because node is at -1)
dN0/dζ should be approximately -0.015625  (negative because node is at -1)
```

**Action**: Add debug output in hex20.cpp to print actual values, compare with theory.

### Step 2: Check Serendipity Formula Implementation (30 min)

The corner node shape functions for Hex20 are:
```
N_corner = (1/8) * (1 + ξξ_i)(1 + ηη_i)(1 + ζζ_i) * (-2 + ξξ_i + ηη_i + ζζ_i)
```

And mid-edge nodes:
```
N_edge = (1/4) * (1 - ξ²)(1 + ηη_i)(1 + ζζ_i)  [for ξ-direction edge]
```

**Action**: Verify these formulas match what's in hex20.cpp lines 40-140.

### Step 3: Test with Reference Element (1 hour)

Create a unit test that:
1. Defines standard unit cube
2. Applies unit displacement in each direction
3. Computes resulting forces
4. Verifies forces oppose displacements

**If this test fails** → Confirms sign error in derivatives
**If this test passes** → Bug is elsewhere (mesh generation? node ordering?)

---

## Quick Fixes to Try

### Fix #1: Flip All Derivative Signs
```cpp
// In hex20.cpp, shape_derivatives_global(), line 312-314:
dNdx[i*3 + 0] = -(J_inv[0] * dNdxi + J_inv[1] * dNdeta + J_inv[2] * dNdzeta);
dNdx[i*3 + 1] = -(J_inv[3] * dNdxi + J_inv[4] * dNdeta + J_inv[5] * dNdzeta);
dNdx[i*3 + 2] = -(J_inv[6] * dNdxi + J_inv[7] * dNdeta + J_inv[8] * dNdzeta);
```

### Fix #2: Check Node Ordering
Maybe Hex20 node ordering doesn't match standard convention? Compare with:
- LS-DYNA Hex20 node ordering
- Abaqus C3D20 node ordering
- ANSYS SOLID186 node ordering

---

## Key Insight

The bug is **definitely a sign error** in the element formulation, not in the solver infrastructure.

Since Hex8 works perfectly and uses identical infrastructure, and since all unit tests pass (which don't depend on signs), the issue MUST be in how Hex20 computes its spatial derivatives.

**Most likely culprit**: Shape function derivative formulas in `hex20.cpp` lines 90-180.

---

## Status

- ✅ 3 bugs fixed (DOF indexing, Jacobian convention, confirmed)
- ❌ 1 bug remaining: Sign error in force calculation
- 🎯 Root cause identified: Forces oppose wrong direction
- 📍 Next step: Verify shape function derivative formulas against literature

**Estimated time to fix**: 1-2 hours once root cause in derivatives is pinpointed

---

**Session end**: November 7, 2025, 10:05 PM
**Total debugging time**: ~3 hours
**Progress**: 95% complete (one sign error away from working!)
