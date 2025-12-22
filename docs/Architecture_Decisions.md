# Architecture Decision Summary
> See `README_NEXTGEN.md` for documentation map and reading order.

## 1. Language & Interop Strategy

- **Decision**: Use C++20 for all new implementation, retain Python for scripting, and encapsulate essential Fortran libraries behind C bindings until equivalent C++ modules meet performance goals.[Unified_Architecture_Blueprint.md:40]
- **Rationale**: Aligns modern tooling and memory-management patterns from RadiossNX while respecting Gemini’s requirement to leverage proven legacy routines during transition.[Unified_Architecture_Blueprint.md:5][Unified_Architecture_Blueprint.md:21][Unified_Architecture_Blueprint.md:24]
- **Open Questions & Resolution**:
  - Prioritised list of Fortran kernels to wrap vs. port outright in Wave 0/1?  
    **Resolution**: Treat implicit solver (`imp_solv.F`), element dispatcher (`forint.F`), and key material models (MAT024, MAT036) as wrap-first targets in Wave 0 to unblock solvers; schedule porting during Waves 2-3 per `Legacy_Migration_Roadmap.md:24`, `Legacy_Migration_Roadmap.md:30`. [Legacy_Migration_Roadmap.md:24][Legacy_Migration_Roadmap.md:30]
  - Criteria (performance delta, maintenance cost) for retiring each Fortran dependency?  
    **Resolution**: Require ≤5% performance gap against Fortran baseline on representative benchmarks plus successful code review sign-off before deprecation; document decisions in architecture records.

## 2. Parallel Runtime & Hardware Abstraction

- **Decision**: Standardise on MPI + Kokkos for distributed and accelerator programming, with OpenMP fallback for CPU-only environments; evaluate task-based execution once baseline is stable.[Unified_Architecture_Blueprint.md:15][Unified_Architecture_Blueprint.md:43]
- **Rationale**: Meets Gemini’s hybrid CPU/GPU requirements and RadiossNX GPU-first direction while minimising divergence across hardware backends.[Unified_Architecture_Blueprint.md:5][Unified_Architecture_Blueprint.md:26][Unified_Architecture_Blueprint.md:28]
- **Open Questions & Resolution**:
  - Which clusters/GPU architectures must be supported in initial releases (CUDA, HIP, SYCL parity)?  
    **Resolution**: Commit to CUDA support (A100, H100) for GA release; provide HIP path (MI250X) as beta using Kokkos portability; SYCL evaluation deferred to Wave 5 optimisation milestone. [Migration_Wave_Assignments.md:3]
  - Do we need CUDA-aware MPI on day one or can we stage it behind a fallback?  
    **Resolution**: Stage CUDA-aware MPI as optional enhancement; initial release uses host-staging fallback with documented performance impact and upgrade path in Wave 5.

## 3. Solver Backend Selection

- **Decision**: Adopt PETSc (with Trilinos as contingency) for distributed sparse solves and built-in GPU backends; supplement with vendor libraries (cuSPARSE/hipSPARSE) for specialised kernels.[Unified_Architecture_Blueprint.md:44]
- **Rationale**: Balances Gemini’s call for industrial-strength linear algebra with RadiossNX’s GPU acceleration ambitions, reducing the need to maintain bespoke solvers early in the project.[Unified_Architecture_Blueprint.md:26][Unified_Architecture_Blueprint.md:58]
- **Open Questions & Resolution**:
  - Which PETSc configurations (precision, solver packages) must be qualified for implicit analyses?  
    **Resolution**: Qualify PETSc double-precision builds with GAMG and KSP/GMRES + ILU preconditioning for structural problems; add single-precision (mixed) tests in GPU pipelines by Wave 3. [Legacy_Migration_Roadmap.md:30]
  - Is Trilinos required for specific physics modules (e.g., MueLu multigrid) that PETSc cannot cover?  
    **Resolution**: Keep Trilinos optional; trigger adoption only if MueLu multigrid yields >20% efficiency gains on coupled thermal-electric benchmarks evaluated during Wave 4.

## 4. I/O & Configuration Formats

- **Decision**: Implement HDF5 and VTK output pipelines with JSON/YAML configuration, and retain adapters for Radioss legacy formats; investigate ADIOS2 for large-scale streaming once baseline is proven.[Unified_Architecture_Blueprint.md:30][Unified_Architecture_Blueprint.md:46]
- **Rationale**: Satisfies interoperability requirements while offering a migration path for existing decks and tooling workflows.[Unified_Architecture_Blueprint.md:12][Unified_Architecture_Blueprint.md:34]
- **Open Questions & Resolution**:
  - Which legacy formats (ANIM, TH, H3D) must be supported at launch vs. provided via converters?  
    **Resolution**: Provide direct ANIM and TH writers at launch; ship H3D converter script leveraging existing reader pipeline in Wave 1 to balance effort. [Legacy_Migration_Roadmap.md:18]
  - Requirements for restart/checkpoint compatibility with existing OpenRadioss jobs?  
    **Resolution**: Support reading legacy restart files via adapter through Wave 3; generate new HDF5-based restart format concurrently and provide conversion utility.

## 5. Interface & Module Structure

- **Decision**: Provide a Python-first API layered over C++ plugins, maintaining CLI compatibility for existing starter workflows and reserving GUI development for post-parity milestones.[Unified_Architecture_Blueprint.md:12]
- **Rationale**: Ensures immediate scripting interoperability and smooth adoption for current users while enabling future UI expansion.[Unified_Architecture_Blueprint.md:5][Unified_Architecture_Blueprint.md:23][Unified_Architecture_Blueprint.md:35]
- **Open Questions & Resolution**:
  - Scope of CLI compatibility (command flags, deck preprocessing) required for early adopters?  
    **Resolution**: Mirror top-level starter flags (`-i`, `-np`, `-restart`) and preprocessing macros in initial CLI layer; capture additional requests via beta feedback forms during Waves 1-2.
  - Governance for third-party plugin integration—how will APIs be versioned and stabilised?  
    **Resolution**: Establish semantic versioning for plugin SDK with LTS support windows; require API review board approval before breaking changes, starting Wave 2.
