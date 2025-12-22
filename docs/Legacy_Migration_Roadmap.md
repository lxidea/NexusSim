# Legacy Migration Roadmap
> See `README_NEXTGEN.md` for documentation map and reading order.

## 1. Objectives

- Port critical OpenRadioss functionality into the unified architecture while maintaining simulation parity and enabling GPU acceleration.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:7][../../OpenRadioss/new_project_spec/specification.md:6][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:19]
- Stage work to mitigate risk around massive Fortran components such as `imp_solv.F` and MPI communication layers.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96][../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:110]

## 2. Migration Waves

### Wave 0 – Enablement (Weeks 0-4)

- **Project skeleton**: Establish C++20/Kokkos codebase, build presets, dependency managers, and testing harness.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:195][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:195]
- **Data schema translation**: Model nodal arrays, element connectivity, and material tables in structure-of-arrays containers to match legacy semantics.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:33][../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:44][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:246]
- **Interop scaffolding**: Set up ISO C binding layer for selective Fortran calls and define C++ strategy interfaces for solver and material components.[../../OpenRadioss/new_project_spec/specification.md:19][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:260]

### Wave 1 – Preprocessing & Mesh (Weeks 4-10)

- **Starter feature parity**: Port mesh ingestion, validation, and domain decomposition logic; wrap existing SPMD routines until full C++ implementation is ready.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:47][../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:170]
- **Input converters**: Implement radioss/LS-DYNA/Abaqus readers using new I/O framework; maintain existing scripts as fallback.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:174][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:154]
- **Mesh services**: Provide mesh partitioning via METIS/ParMETIS and set up ghost layer management consistent with legacy behaviour.[../../OpenRadioss/new_project_spec/specification.md:167][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:129]

### Wave 2 – Explicit Solver Core (Weeks 8-18)

- **Element kernels**: Reimplement or wrap `forint.F` dispatch and representative element formulations (shell, solid, beam) with GPU-capable kernels.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:60][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:256]
- **Time integration loop**: Port explicit integrators from `radioss2.F`/`resol.F` into modular C++ strategy classes.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:56][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:260]
- **Contact mechanics**: Introduce penalty/lagrange contact modules, initially delegating to Fortran routines; plan GPU variants for collision detection.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:73][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:57]

### Wave 3 – Implicit Solver Suite (Weeks 16-32)

- **Modular decomposition**: Split `imp_solv.F` into nonlinear solver, stiffness assembly, integrator, and linear solver components with clear interfaces.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:260]
- **Solver backend integration**: Connect PETSc/Trilinos for distributed solves; prototype custom GPU solvers where needed.[../../OpenRadioss/new_project_spec/specification.md:187][../../viRadioss/PROJECT_SUMMARY.md:76]
- **Convergence & diagnostics**: Implement residual monitors and logging consistent with Gemini’s robustness requirements.[../../OpenRadioss/new_project_spec/specification.md:9][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:201]

### Wave 4 – Multi-Physics & Coupling (Weeks 24-40)

- **Coupling framework**: Implement observer-based field exchange and explicit/implicit coupling modes to satisfy Multi-physics requirements.[../../OpenRadioss/new_project_spec/specification.md:22][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:301]
- **Meshfree integration**: Incorporate SPH, DEM, and bridge domain methods leveraging RadiossNX module layout while validating against legacy SPH modules.[../../OpenRadioss/new_project_spec/specification.md:6][../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:78]
- **Thermal/EM modules**: Add secondary physics plugins with appropriate solver strategies and coupling hooks.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:57]

### Wave 5 – Optimisation & Decommissioning (Weeks 36+)

- **GPU optimisation**: Profile kernels, optimise memory arenas, and adopt task-based scheduling where beneficial.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:33][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:215]
- **Legacy retirement**: Replace remaining Fortran components once equivalent C++ implementations meet performance and validation criteria.
- **Documentation & training**: Update onboarding materials, cross-reference module maps, and finalise architecture decision records.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:45]

## 3. Risk Mitigation

- **Parallel performance regressions**: Maintain MPI+OpenMP benchmarks during each wave and compare against OpenRadioss baselines.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96]
- **Validation coverage**: Convert QA decks into automated tests to guard against behavioural drift.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:186][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:167]
- **Resource constraints**: Allocate specialist time for implicit solver refactor due to size and complexity of legacy code (366k lines).[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96]

## 4. Deliverables Per Wave

- **Wave 0**: Build system, data containers, interop harness ready.
- **Wave 1**: Mesh pipeline operational with partitioning and ghost management.
- **Wave 2**: Explicit solver executing benchmark cases on CPU/GPU.
- **Wave 3**: Implicit solver available with distributed linear algebra backend.
- **Wave 4**: Coupled multi-physics scenarios validated.
- **Wave 5**: Performance tuned GPU kernels, retire legacy dependencies, complete documentation set.
