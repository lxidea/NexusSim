# Changelog

All notable changes to the NexusSim project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-19

### Added

#### Phase 1: Core Framework and Foundational Capabilities

- **Wave 1 — Core Materials (14 models):** Elastic, Von Mises, Johnson-Cook, Neo-Hookean,
  Mooney-Rivlin, Ogden, orthotropic, piecewise-linear, tabulated, foam, crushable foam,
  honeycomb, viscoelastic, Cowper-Symonds, Zhao, elastic-plastic-fail, rigid, null.
  61 test assertions.

- **Wave 2 — Failure Models (6 models):** Hashin, Tsai-Wu, Chang-Chang, GTN, GISSMO,
  tabulated failure. 52 test assertions.

- **Wave 3 — Rigid Bodies and Constraints:** Rigid body dynamics (Newton-Euler), kinematic
  constraints (RBE2, RBE3, revolute/spherical/cylindrical joints), rigid walls (planar,
  cylindrical, spherical, moving). 50 test assertions.

- **Wave 4 — Loads and Initial Conditions:** Load curves with extrapolation modes, load
  manager (nodal force, body force, pressure, prescribed motion), initial conditions
  (velocity, temperature). 46 test assertions.

- **Wave 5 — Tied Contact and Equations of State:** Tied contact with failure, 5 EOS
  models (ideal gas, Mie-Gruneisen, JWL, polynomial, tabulated). 34 test assertions.

- **Wave 6 — Checkpoint/Restart and Enhanced Output:** Basic and extended checkpoint I/O,
  time history, part energy, interface force, cross-section force, result database output.
  100 test assertions.

- **Wave 7 — Composite Analysis:** Classical Lamination Theory, ABD matrix, thermal
  residual stress, interlaminar shear, progressive failure (FPF, ply degradation, strength
  envelope). 79 test assertions.

- **Wave 8 — Sensors, Controls, and ALE:** 5 sensor types with CFC filtering, 8 control
  action types, ALE mesh management (Laplacian/weighted smoothing, donor-cell/Van Leer
  advection). 111 test assertions.

#### Phase 2: Capability Expansion (Waves 9-22)

- **Wave 9 — Explicit Dynamics:** Central difference time integration, bulk viscosity,
  hourglass control (Flanagan-Belytschko), energy monitoring, element erosion. 114 test
  assertions.

- **Wave 10 — Advanced Materials (20 models):** Hill anisotropic, Barlat Yld2000,
  tabulated Johnson-Cook, concrete (Drucker-Prager cap), fabric, cohesive zone, soil cap,
  user-defined, Arruda-Boyce, shape memory alloy, rate-dependent foam, Prony viscoelastic,
  thermal elastic-plastic, Zerilli-Armstrong, Steinberg-Guinan, MTS, Blatz-Ko with Mullins,
  laminated glass, spot weld, rate-dependent composite. 128 test assertions.

- **Wave 11 — Extended Failure Models (12 models):** Johnson-Cook failure,
  Cockcroft-Latham, Lemaitre CDM, Puck, FLD, Wilkins, Tuler-Butcher, maximum stress,
  maximum strain, energy-based, Wierzbicki (modified Mohr-Coulomb), fabric failure. 101
  test assertions.

- **Wave 12 — Contact Expansion (9 types):** Contact stiffness scaler, velocity-dependent
  friction, shell thickness contact, edge-to-edge, segment-based, self-contact, symmetric,
  rigid-deformable, multi-surface contact manager. 60 test assertions.

- **Wave 13 — Extended EOS (8 models):** Murnaghan, Noble-Abel, stiff gas (Tait),
  Tillotson, Sesame, PowderBurn, compaction, Osborne. 49 test assertions.

- **Wave 14 — Thermal Solver:** Heat conduction (explicit forward-Euler), convection BC,
  radiation BC, fixed temperature BC, heat flux BC, adiabatic heating (Taylor-Quinney),
  thermal time step control, coupled thermo-mechanical. 49 test assertions.

- **Wave 15 — Extended Elements (7 types):** ThickShell8, ThickShell6, DKT shell, DKQ
  shell, plane element, axisymmetric element, connector element. 80 test assertions.

- **Wave 16 — Advanced Capabilities (8 modules):** Modal analysis (Lanczos eigensolver),
  XFEM (level-set crack, Heaviside/branch enrichment), CONWEP blast (Kingery-Bulmash),
  airbag simulation, seatbelt dynamics, advanced ALE (Euler solver, cut-cell, turbulence),
  adaptive mesh refinement (ZZ error estimator), draping analysis. 77 test assertions.

- **Wave 17 — MPI Completion (6 components):** Distributed stiffness assembly (CSR),
  ghost node communication (async), domain decomposition (RCB, weighted greedy), parallel
  contact detection (two-phase AABB), load balancing (greedy diffusion), scalability
  benchmarking. 82 test assertions.

- **Wave 18 — Tier 2 Materials (20 models):** Explosive burn, porous elastic, brittle
  fracture, creep, kinematic hardening, Drucker-Prager, tabulated composite, ply
  degradation, orthotropic plastic, pinching, frequency-dependent viscoelastic,
  generalized viscoelastic, phase transformation, polynomial hardening, viscoplastic
  thermal, porous brittle, anisotropic crush foam, spring hysteresis, programmed
  detonation, bonded interface. 126 test assertions.

- **Wave 19 — Failure Models and SPH Enrichment:** 10 failure models (LaDeveze
  delamination, Hoffman, Tsai-Hill, RTCl, Mullins, spalling, HC_DSSE, adhesive joint,
  windshield, generalized energy) + 7 SPH enrichment features (tensile instability fix,
  multi-phase, boundary handling, thermal SPH, MUSCL SPH, particle shifting, SPH damage).
  215 test assertions.

- **Wave 20 — Production I/O and Elements:** 6 output formats (binary animation, H3D,
  D3PLOT, EnSight Gold, time history, cross-section force) + 6 element formulations
  (Belytschko-Tsay shell, Pyramid5, MITC4, EAS Hex8, B-bar Hex8, isogeometric shell).
  113 test assertions.

- **Wave 21 — ALE and Contact Refinements:** 6 ALE features (FVM advection, MUSCL
  reconstruction, 2D ALE, multi-fluid VOF, ALE-FSI coupling, ALE remapping) + 6 contact
  refinements (automatic detection, 2D contact, SPH contact, airbag fabric, contact heat
  generation, mortar friction). 114 test assertions.

- **Wave 22 — Input Readers, Preprocessing, and Solver Hardening:** 4 input readers
  (Radioss D00, LS-DYNA keyword extended, ABAQUS INP, model validator) + 5 preprocessing
  utilities (mesh quality, mesh repair, auto contact surface, material assignment,
  coordinate transform) + 5 solver features (mass scaling, subcycling, added mass
  monitoring, dynamic relaxation, smooth contact). 132 test assertions.

#### Phase 3: Infrastructure and Release Preparation

- **Wave 23 — CTest Integration:** 113 test executables registered with CTest, test
  labeling by category, `ctest --output-on-failure` support.

- **Wave 24 — Code Quality Tooling:** GitHub issue/PR templates, clang-format
  configuration, contribution guidelines.

- **Wave 25 — CI/CD Pipeline:** GitHub Actions workflow for build, test, and static
  analysis on push/PR.

- **Wave 26 — Enum Consolidation and CMake Restructuring:** Unified MaterialType enum,
  cmake/TestTargets.cmake modularization.

- **Wave 27 — Documentation Refresh:** Sphinx manual updated to reflect all 22 waves,
  accurate capability counts (54 materials, 28 failure models, 13 EOS, 23 elements,
  22 contact types), CHANGELOG.md.

### Project Statistics

| Metric | Count |
|--------|-------|
| Material models | 54 |
| Failure/damage models | 28 |
| Equations of state | 13 |
| Element formulations | 23 |
| Contact types | 22 |
| Header files | 145 |
| Lines of C++20/Kokkos code | 73,000+ |
| Test assertions | 2,800+ |
| CTest-registered executables | 113 |
| Passing tests | 108 |
| Known-fail tests | 5 |

[0.1.0]: https://github.com/nexussim/nexussim/releases/tag/v0.1.0
