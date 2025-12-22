# Unified Architecture Blueprint
> See `README_NEXTGEN.md` for documentation map and reading order.

## 1. Vision Alignment

- **Functional scope**: Maintain OpenRadioss starter/engine/common workflow, extending to multi-physics, FEM–meshfree coupling, and hybrid HPC targets stated in the Gemini specification.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:7][../../OpenRadioss/new_project_spec/specification.md:6][../../OpenRadioss/new_project_spec/specification.md:30]
- **Modern execution**: Adopt RadiossNX layered architecture and plugin-based modules to achieve GPU-first performance and maintainability.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:17][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:33][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:45]

## 2. Layered System Structure

| Layer | Legacy Responsibilities | RadiossNX Proposal | Unified Approach |
| --- | --- | --- | --- |
| **Interface** | CLI-driven workflows via starter inputs.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:9] | Python/C++ APIs, GUI, web clients.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:47] | Expose Python API for scripting and interop; provide CLI compatibility layer for existing decks; plan optional GUI after core parity. |
| **Application (Physics)** | Starter/engine handle all physics in monolithic Fortran modules.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:7][../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:11] | Plugin physics modules (solid, fluid, thermal, EM, contact, fracture, FSI).[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:57][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:62] | Define physics packages with C++20 plugin interfaces; wrap legacy Fortran models via interop layer until fully reimplemented. |
| **Computational Framework** | Shared modules in `common_source` deliver data structures, MPI utilities, solvers, and I/O.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:32][../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:44] | Distinct subsystems for mesh, materials, solvers, contact, I/O, tests.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:129] | Introduce C++ core framework implementing module tree; progressively port Fortran modules into C++ services while retaining interoperable access. |
| **Runtime & Hardware** | Hybrid MPI+OpenMP with CPU focus; optional mass scaling.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96][../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:110] | MPI + GPU (CUDA/HIP/SYCL) and task-based concurrency.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:19][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:33][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:256] | Target hybrid MPI + GPU execution using Kokkos; preserve OpenMP fallback; ensure CPU-only path remains for portability. |

## 3. Module Decomposition

### 3.1 Core Packages

- **Mesh & State Management**
  - Preserve data semantics from `nodal_arrays.F90` and `connectivity.F90` within a C++ structure-of-arrays layout for GPU portability.[../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:33][../../OpenRadioss/gemini_analysis/OpenRadioss_Analysis.md:44][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:246]
- **Physics Modules**
  - Solid, fluid, thermal, EM, contact, fracture, and FSI packages implemented as plugins with well-defined interfaces; initial versions wrap legacy routines where reuse is mandatory.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:57][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:62]
- **Solver Suite**
  - Explicit and implicit solvers modularised; implicit functionality refactored from `imp_solv.F` into smaller C++ components exposed via strategy pattern.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:260]
- **Coupling Framework**
  - Layer providing field registries, observer-based exchange, and configuration of explicit vs. implicit coupling modes.[../../OpenRadioss/new_project_spec/specification.md:22][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:301]
- **I/O and Data Management**
  - Support HDF5/VTK/Exodus for results, maintain Radioss formats for backward compatibility via reader/writer adapters.[../../OpenRadioss/new_project_spec/specification.md:34][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:154]

### 3.2 Support Packages

- **Pre/Post Tools**: Replace starter preprocessing with modular import pipeline (RAD, LS-DYNA, Abaqus readers) plus mesh conditioning utilities.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:174][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:154]
- **Testing & QA**: Integrate Catch2 unit tests and Python regression harness; migrate legacy QA decks into automated suites.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:167]
- **Utilities**: Provide profiling hooks, memory arenas, logging, and error handling consistent with RadiossNX design guidelines.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:201][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:215]

## 4. Technology Stack Decisions

| Concern | Gemini Preference | RadiossNX Preference | Unified Decision |
| --- | --- | --- | --- |
| **Languages** | C++17/20 core with selective Fortran; Python scripting.[../../OpenRadioss/new_project_spec/specification.md:178] | C++20 + Python, no Fortran in core.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:21] | Adopt C++20 for new code; encapsulate Fortran libraries behind C bindings where performance reuse is justified; plan gradual retirement. |
| **Parallelism** | MPI + OpenMP + GPU kernels (CUDA/HIP).[../../OpenRadioss/new_project_spec/specification.md:30][../../OpenRadioss/new_project_spec/specification.md:161] | MPI + GPU with task-based concurrency (Kokkos, SYCL option).[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:19][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:33] | Implement MPI + Kokkos (CUDA/HIP/SYCL) plus OpenMP fallback; evaluate task runtime as future enhancement. |
| **Linear Algebra** | PETSc / Trilinos for sparse solvers.[../../OpenRadioss/new_project_spec/specification.md:187] | Eigen, cuBLAS/cuSPARSE, custom GPU solvers.[../../viRadioss/PROJECT_SUMMARY.md:76] | Use PETSc as primary distributed solver with GPU backends (via Kokkos); leverage cuSPARSE/hipSPARSE for custom kernels where PETSc insufficient. |
| **Dependency Management** | CMake-based.[../../OpenRadioss/new_project_spec/specification.md:197] | CMake + Conan/vcpkg, Ninja, ccache.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:195] | Adopt CMake presets with Conan for third-party packages; support vcpkg alternative; standardise on Ninja + ccache for dev builds. |
| **I/O Formats** | HDF5, VTK, JSON/YAML.[../../OpenRadioss/new_project_spec/specification.md:193] | HDF5, ADIOS2, VTK, legacy support.[../../viRadioss/PROJECT_SUMMARY.md:84] | Provide HDF5 + VTK as mandatory; evaluate ADIOS2 for HPC streaming; maintain JSON/YAML configs and Radioss format adapters. |

## 5. Implementation Phases

1. **Core Framework Initialization**
   - Scaffold C++20 project with module tree, build tooling, and testing harness.
   - Implement mesh/state SOA containers backed by Kokkos views.
2. **Legacy Interop Layer**
   - Wrap critical Fortran kernels (implicit solvers, material laws) using ISO C bindings; expose through strategy interfaces.
3. **Physics Plugin Development**
   - Port explicit solid mechanics first; subsequently add fluid, contact, and FSI modules, ensuring observer-based coupling APIs.
4. **GPU Kernel Enablement**
   - Translate high-impact loops (element forces, SPH neighbor search) into Kokkos kernels; measure parity with CPU path.
5. **I/O & Tooling**
   - Implement HDF5/VTK writers and configuration parsers; add CLI compatibility mode for legacy decks.
6. **Validation & QA**
   - Rebaseline regression suite using automated tests; monitor performance vs. OpenRadioss benchmarks.

## 6. Governance & Documentation

- Maintain architecture decision records capturing migration of Fortran assets, GPU kernel adoption, and solver choices.
- Provide developer onboarding guides derived from key routine inventory and plugin interfaces.
