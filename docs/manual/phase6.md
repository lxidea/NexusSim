# Phase 6: Full OpenRadioss Parity + MPI Production

**Waves 39--45** | **Goal:** Achieve 100% algorithmic coverage and multi-node production readiness

---

## Overview

Phase 6 closes the remaining gaps between NexusSim and the OpenRadioss reference
implementation. It delivers the final material laws, completes implicit solver
hardening, adds production-depth output and starter features, deepens XFEM and
airbag models, and wraps up with a cross-wave validation suite and full
documentation.

After Phase 6, NexusSim covers:

| Category               | NexusSim | OpenRadioss | Coverage |
|------------------------|----------|-------------|----------|
| Material models        | 115      | 115         | 100%     |
| Failure models         | 42       | 42          | 100%     |
| Equations of state     | 20       | 20          | 100%     |
| Element formulations   | 35       | 35          | 100%     |
| Contact types          | 37       | 37          | 100%     |
| Solver capabilities    | 15+      | 15+         | 100%     |
| Output formats         | 12       | 12          | 100%     |
| Input readers          | 4        | 4           | 100%     |
| MPI subsystems         | 5        | 5           | ~95%     |

---

## Wave 39 -- Final Material Laws + Implicit Solver Hardening

### 39a: Six Material Models

| # | Class                        | LAW | Description                                          |
|---|------------------------------|-----|------------------------------------------------------|
| 1 | `DPCapMaterial`              | 81  | Drucker-Prager with elliptical cap hardening         |
| 2 | `ThermalMetallurgyMaterial`  | 80  | Phase transformation kinetics (JMAK + Koistinen-Marburger) |
| 3 | `ElasticShellMaterial`       | 4   | Shell-specific elastic-plastic with thickness update |
| 4 | `CompositeDamageMaterial`    | 53  | Progressive composite damage (Hashin criteria)       |
| 5 | `NonlinearElasticMaterial`   | 44  | Nonlinear elastic polynomial/exponential (reversible)|
| 6 | `MohrCoulombMaterial`        | 27  | Classic Mohr-Coulomb with tension cutoff             |

Key features:

- **DPCapMaterial** uses a two-surface formulation: a linear Drucker-Prager shear
  cone plus an elliptical cap that hardens with plastic volumetric strain. Cap
  position `pb` evolves via the compaction curve `W * (1 - exp(-D * |eps_v_p|)) / D`.
- **ThermalMetallurgyMaterial** tracks five metallurgical phases (austenite,
  ferrite, pearlite, bainite, martensite). Diffusive phases follow JMAK kinetics;
  martensite uses the Koistinen-Marburger athermal transformation.
- **CompositeDamageMaterial** implements Hashin failure criteria with four
  independent damage variables: fiber tension (d1), fiber compression (d2),
  matrix tension (d3), matrix compression (d4).

### 39b: Six Implicit Solver Features

| # | Class                  | Description                                          |
|---|------------------------|------------------------------------------------------|
| 1 | `MUMPSSolver`          | Sparse direct LDL^T with Bunch-Kaufman pivoting      |
| 2 | `ImplicitBFGS`         | L-BFGS quasi-Newton minimisation                     |
| 3 | `ImplicitBuckling`     | Linear buckling eigenvalue (inverse iteration)       |
| 4 | `ImplicitDtControl`    | Adaptive time step (convergence ratio based)         |
| 5 | `IterativeRefinement`  | Iterative refinement for ill-conditioned systems     |
| 6 | `ImplicitContactK`    | Penalty contact stiffness assembly                   |

---

## Wave 40 -- Output Format Completion + Starter Parity

### 40a: Six Output Format Writers

| # | Class                | Description                                 |
|---|----------------------|---------------------------------------------|
| 1 | `RadiossAnimWriter`  | Native RADIOSS `.anim` binary animation     |
| 2 | `StatusFileWriter`   | ASCII `.sta` status file (energy, mass err) |
| 3 | `DynainWriter`       | Dynain restart file (deformed mesh + state) |
| 4 | `ReactionForcesTH`   | Reaction forces time history extraction     |
| 5 | `QAPrintWriter`      | Quality assurance print (mesh metrics)      |
| 6 | `ReportGenerator`    | Simulation summary report (ASCII)           |

### 40b: Five Starter Features

| # | Class                | Description                                 |
|---|----------------------|---------------------------------------------|
| 7 | `RadiossStarterFull` | Complete D00 keyword parsing                |
| 8 | `PropertyReader`     | `/PROP` card reader (shell, beam, solid)    |
| 9 | `SectionForceOutput` | Cross-section force/moment extraction       |
| 10| `ModelValidatorExt`  | Extended model validation                   |
| 11| `ErrorMessageSystem` | Structured error/warning/info messages      |

---

## Wave 41 -- Production Depth: XFEM, Airbag, Coupling, Acoustics

### 41a: XFEM Production Hardening (4 features)

| # | Class                | Description                                 |
|---|----------------------|---------------------------------------------|
| 1 | `XFEMFatigueCrack`   | Paris-law fatigue crack growth + cycle count|
| 2 | `XFEMMultiCrack`     | Multiple cracks with shielding + coalescence|
| 3 | `XFEMAdaptiveMesh`   | h-adaptive refinement (Zienkiewicz-Zhu)     |
| 4 | `XFEMOutputFields`   | Crack path + SIF + COD extraction, VTK out  |

### 41b: Airbag Production Hardening (4 features)

| # | Class                  | Description                               |
|---|------------------------|-------------------------------------------|
| 5 | `AirbagMultiChamber`   | Multi-chamber with orifice + check valves |
| 6 | `AirbagGasSpecies`     | Multi-species gas with mixing rules       |
| 7 | `AirbagTTF`            | Tank test format inflator data import     |
| 8 | `AirbagMembraneDrape`  | Gravity draping with fold detection       |

### 41c: Coupling + Acoustics Hardening (4 features)

| # | Class                       | Description                          |
|---|-----------------------------|--------------------------------------|
| 9 | `CouplingSubIteration`      | Implicit FSI with Aitken relaxation  |
| 10| `CouplingFieldSmoothing`    | Laplacian smoothing of interface     |
| 11| `AcousticFMM`               | Fast Multipole Method for BEM        |
| 12| `AcousticStructuralModes`   | Acoustic-structural modal coupling   |

---

## Wave 42 -- Validation + Documentation

Wave 42 is the final wave. It creates a single cross-wave validation test
(`parity_wave42_test`) that exercises every major subsystem introduced in
Waves 39--41, and produces three documentation files:

| Deliverable              | File                              | Content                            |
|--------------------------|-----------------------------------|------------------------------------|
| Parity validation test   | `examples/parity_wave42_test.cpp` | ~100 tests across 6 scenarios      |
| Phase 6 documentation    | `docs/manual/phase6.md`           | This file                         |
| Material catalog         | `docs/manual/material_catalog.md` | All 115 models with LAW IDs        |
| Migration guide          | `docs/manual/migration_guide.md`  | OpenRadioss-to-NexusSim migration  |

### Validation Scenarios

1. **MaterialLawParity** (~30 tests) -- All six Wave 39 materials under uniaxial
   tension, hydrostatic compression, and special loading (cap engagement, phase
   transformation, tension cutoff).
2. **FailureParity** (~15 tests) -- Damage variable growth, cap hardening
   evolution, yield state transitions, shell thickness reduction.
3. **ImplicitBenchmark** (~15 tests) -- MUMPS direct solve, L-BFGS convergence,
   buckling eigenvalue, adaptive time step, iterative refinement, penalty contact.
4. **EulerShockTube** (~15 tests) -- All six Wave 40 output writers validated via
   roundtrip I/O, plus the error message system.
5. **AirbagBenchmark** (~15 tests) -- Multi-chamber equilibration, gas species
   mixing, fatigue crack growth, multi-crack shielding, TTF interpolation,
   membrane draping.
6. **ContactBenchmark** (~10 tests) -- FSI sub-iteration convergence, field
   smoothing, FMM acoustic pressure, acoustic-structural mode coupling.

---

## LAW Cross-Reference (Wave 39 Materials)

| NexusSim Class                  | OpenRadioss LAW | Category           |
|---------------------------------|-----------------|--------------------|
| `DPCapMaterial`                 | LAW81           | Soil/Geomechanical |
| `ThermalMetallurgyMaterial`     | LAW80           | Thermal/Metallurgy |
| `ElasticShellMaterial`          | LAW4            | Shell Metals       |
| `CompositeDamageMaterial`       | LAW53           | Composites         |
| `NonlinearElasticMaterial`      | LAW44           | Hyperelastic       |
| `MohrCoulombMaterial`           | LAW27           | Soil/Geomechanical |

---

## Wave 43 -- Gap Closure

Wave 43 closes the remaining algorithmic gaps identified during the Wave 42
validation pass: EOS models, load types, contact search, shell improvements,
constraints, multiphysics boundaries, per-element assembly, and output extractors.

### 43a: Seven EOS Models

| # | Class                 | Description                                     |
|---|-----------------------|-------------------------------------------------|
| 1 | `LSZKEOS`             | Lee-Tarver (LSZK) reactive burn EOS             |
| 2 | `NASGEOS`             | Noble-Abel Stiffened Gas                        |
| 3 | `PuffEOS`             | Puff three-phase with spall                     |
| 4 | `ExponentialEOS`      | Exponential p = A*exp(-R1*V) + B*exp(-R2*V)     |
| 5 | `IdealGasVTEOS`       | Ideal gas (V,T) formulation                     |
| 6 | `Compaction2EOS`      | Two-phase compaction (powder metals)            |
| 7 | `CompactionTabEOS`    | Tabulated compaction with interpolation         |

### 43b: Five Load Types

| # | Class                   | Description                                  |
|---|-------------------------|----------------------------------------------|
| 1 | `CentrifugalLoad`       | Rotating body centrifugal + Coriolis forces  |
| 2 | `CylindricalPressure`   | Pressure on cylindrical/spherical surfaces   |
| 3 | `FluidLoad`             | Hydrostatic + hydrodynamic fluid loads       |
| 4 | `LaserLoad`             | Laser ablation pressure (Gaussian profile)   |
| 5 | `BoltPreload`           | Bolt preload with cross-section control      |

### 43c: Contact & Element Enhancements

- **BucketSort3D**: Uniform spatial hash grid for O(N) broad-phase contact search
- **AABB tree**: Bounding volume hierarchy for narrow-phase pruning
- **ContactSortManager**: Orchestrates multi-stage search pipeline
- **Shell warp correction**: Warp detection, drilling DOF stabilization, hourglass control
- **FXBODY/RLINK/GroupSet**: Superelement, velocity link, and group algebra constraints
- **EBCS + joints + springs**: Valve/propellant/non-reflecting boundaries, universal/planar/translational joints

### 43d: Per-Element Assembly + Output Extractors

- **Per-element assemblers**: Hex8/20, Tet4/10, Shell3/4, Beam2, Spring — each with
  stiffness, mass, and internal force assembly
- **Element buffer system**: Pre-allocated per-type buffers for contiguous memory access
- **Per-entity output extractors**: Node, shell, solid, SPH, beam, rigid body,
  interface, crack, and cross-section extractors (148 tests)

---

## Wave 44 -- Production Depth: Contact, TH, Sensors, Spot Weld, Tuning

- **ContactGap**: Improved gap calculation with edge-to-edge and tied sliding
- **RadiossTHWriter**: Native time history binary format output
- **SensorExpression**: Expression-based sensor evaluation with CFC filtering
- **SpotWeldElement**: Hex and beam-based spot weld with failure criteria
- **TuningConstants**: Hourglass modes (IHQ1-8), drilling penalty calibration,
  bulk viscosity coefficients

---

## Wave 45 -- MPI Production + Final Tuning

Wave 45 transforms NexusSim from single-node to multi-node production. It
fixes the CMake MPI build, adds a comprehensive test infrastructure, and
implements the five major MPI subsystems matching OpenRadioss patterns.

### 45a: MPI Build Fix + Test Infrastructure

- **CMake fix**: Removed hardcoded `MPI_HOME`; system MPICH auto-detected by
  `find_package(MPI)`
- **`nexussim_add_mpi_test` macro**: Runs tests under `mpiexec -n N`
- **MPITestHarness**: RAII wrapper for `MPI_Init_thread` / `MPI_Finalize`
- **MPIAssert**: Collective assertions via `MPI_Allreduce(MPI_LAND)`
- **MPITestRunner**: Orchestrates SPMD test execution with pass/fail tracking

### 45b: Production Force Exchange

| # | Class                   | Description                                     |
|---|-------------------------|-------------------------------------------------|
| 1 | `FrontierPattern`       | Per-neighbor send/recv frontier indices + buffers (OpenRadioss IAD_ELEM equivalent) |
| 2 | `ForceExchanger`        | Non-blocking ghost-to-owner force accumulation + owner-to-ghost scatter |
| 3 | `DistributedTimeStep`   | Global dt via `MPI_Allreduce(MPI_MIN)`          |
| 4 | `DistributedEnergyMonitor` | Global energy summation via `MPI_Allreduce(MPI_SUM)` |

### 45c: Parallel Contact Search

| # | Class                   | Description                                     |
|---|-------------------------|-------------------------------------------------|
| 1 | `DistributedBroadPhase` | Multi-stage: rank AABB allgather → overlap test → element AABB exchange → cross-rank pairs |
| 2 | `ContactDataExchanger`  | Non-blocking node position/velocity exchange for narrow-phase |
| 3 | `ParallelBucketSort`    | Distributed bucket sort with halo exchange      |

### 45d: Parallel I/O + Load Rebalancing

| # | Class                   | Description                                     |
|---|-------------------------|-------------------------------------------------|
| 1 | `OutputGatherer`        | Rank-0 centralization via `MPI_Gatherv`         |
| 2 | `ParallelAnimWriter`    | Gather → root writes `.anim` frames            |
| 3 | `ParallelTHWriter`      | Gather → root writes time history              |
| 4 | `MigrationExecutor`     | Element data pack/send/unpack for load rebalancing |
| 5 | `DynamicRepartitioner`  | Imbalance detection → repartition → rebuild    |

### 45e: Final Tuning + Scalability Validation

| # | Class                        | Description                                |
|---|------------------------------|--------------------------------------------|
| 1 | `AdaptiveHourglassSelector`  | Mesh-quality-driven IHQ mode selection     |
| 2 | `ContactStabilizationDamper` | Critical damping for contact interfaces    |
| 3 | `ElementTimeStepCalibrator`  | Per-element-type safety factors            |
| 4 | `MPIScalabilityValidator`    | Strong scaling + comm overhead validation  |
| 5 | `ProductionMPIDriver`        | Full production cycle orchestrator         |

---

## Project Totals After Phase 6

| Metric                 | Count  |
|------------------------|--------|
| Total tests            | ~5,570 |
| Material models        | 115    |
| Failure models         | 42     |
| Equations of state     | 20     |
| Element formulations   | 35     |
| Contact types          | 37     |
| Solver features        | 15+    |
| Output format writers  | 12     |
| Input readers          | 4      |
| MPI subsystems         | 5      |
| Completed waves        | 45     |
| Algorithmic parity     | ~100%  |
| Tuning parity          | ~90%   |
| MPI parity             | ~95%   |
