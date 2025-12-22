# NexusSim Development TODO

**Quick Reference**: Active development priorities
**Detailed Version**: See `docs/TODO.md` for comprehensive task list
**Last Updated**: 2025-11-08

---

## 🔥 CRITICAL (This Week)

- [ ] **Fix Hex20 Force Sign Bug** (1-2 hours)
  - Root cause: Shape function derivatives have wrong sign
  - Location: `src/discretization/fem/solid/hex20.cpp` lines 90-180, 312-314
  - Action: Verify formulas against Hughes FEM textbook, fix sign
  - Success: Bending test runs without NaN

- [ ] **Verify GPU Backend** (30 min)
  - Check: `cmake .. -LAH | grep -i kokkos` shows `CUDA=ON`
  - If not: Rebuild with `-DKokkos_ENABLE_CUDA=ON`
  - Success: GPU execution confirmed

- [ ] **Run GPU Benchmarks** (2-4 hours)
  - Run: `./bin/gpu_performance_benchmark`
  - Measure: CPU vs GPU speedup (target: 50-100x)
  - Success: Performance documented

---

## 📋 IMPORTANT (Next 2-4 Weeks)

- [ ] **Update Documentation** (2-4 hours)
  - Roadmap: Element library 30% → 85%, GPU 0% → 80%
  - Files: `README.md`, `docs/Development_Roadmap_Status.md`, `docs/WHATS_LEFT.md`

- [ ] **Implement Material Models** (1-2 weeks)
  - [ ] Johnson-Cook plasticity (5 days)
  - [ ] Neo-Hookean hyperelastic (3 days)

- [ ] **Add Mesh Validation** (3-5 days)
  - Topology checks, positive Jacobian, no orphans
  - Location: `src/data/mesh_validator.cpp` (new)

- [ ] **Expand Test Coverage** (1 week)
  - Integrate Catch2 framework
  - Convert examples to unit tests
  - Setup CI/CD pipeline

---

## 🔧 MEDIUM PRIORITY (1-3 Months)

- [ ] **Radioss Format Reader** (1-2 weeks)
  - Legacy compatibility for OpenRadioss migration

- [ ] **Contact Mechanics** (2-3 weeks)
  - Penalty contact phase 1
  - Friction models phase 2

- [ ] **Implicit Solver** (2-3 months)
  - Newmark-β integrator
  - Newton-Raphson solver
  - PETSc integration

---

## 📊 NOTES

**Current Status** (2025-11-08):
- ✅ 6/7 elements production-ready (Hex8, Tet4, Tet10, Shell4, Wedge6, Beam2)
- ⚠️ Hex20 has force sign bug (95% ready)
- ✅ GPU kernels 80% implemented
- ✅ Architecture 100% aligned with goals

**Timeline to v1.0 Beta**: 4 weeks (if prioritized)

**Blockers**: Only Hex20 bug (1-2 hours to fix)

---

## 📚 Reference Documents

- `docs/TODO.md` - Detailed task list with fix paths
- `docs/PROGRESS_VS_GOALS_ANALYSIS.md` - Comprehensive progress analysis
- `docs/KNOWN_ISSUES.md` - Bug tracking
- `docs/ELEMENT_LIBRARY_STATUS.md` - Element status
- `DEVELOPMENT_REFERENCE.md` - Feature planning guide

---

*See `docs/TODO.md` for detailed implementation notes and success criteria*
