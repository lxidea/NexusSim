# FSI Coupling Prototype Plan
> See `README_NEXTGEN.md` for documentation map and reading order.

## 1. Objective

Validate the coupling framework and GPU execution pathway by running a representative fluid–structure interaction benchmark, exercising explicit coupling, field registries, and device-aware transfers defined in the coupling specification.[Coupling_GPU_Specification.md:5][Coupling_GPU_Specification.md:19][Coupling_GPU_Specification.md:30]

## 2. Scenario Selection

- **Benchmark**: 2D flexible plate in channel flow (FSI3) with moderate Reynolds number—well-documented reference solution for displacement and drag.
- **Domains**:
  - **Solid**: Clamped elastic plate (FEM shell element set) using explicit structural solver.[Coupling_GPU_Specification.md:28][Coupling_GPU_Specification.md:35]
  - **Fluid**: Incompressible ALE formulation discretised on structured mesh with GPU-enabled solver kernels.[Coupling_GPU_Specification.md:28][Coupling_GPU_Specification.md:36]
- **Coupling**: Explicit staggered scheme with displacement-to-mesh-motion and traction-to-force operators on GPU.

## 3. Work Breakdown

| Step | Description | Owners | Dependencies |
| --- | --- | --- | --- |
| 1 | Implement FieldRegistry entries for structural displacement, fluid traction, and mesh motion; ensure DualView usage for CPU fallback. *(status: design ready per `FSI_Field_Registration.md`)*[Coupling_GPU_Specification.md:19][Coupling_GPU_Specification.md:35][FSI_Field_Registration.md:1] | Multi-Physics team (E. Gomez) | Wave 0 data containers |
| 2 | Develop GPU-enabled CouplingOperator pair (disp → mesh, traction → force) with Catch2 unit tests covering analytic transfer cases.[Coupling_GPU_Specification.md:20][Coupling_GPU_Specification.md:58] | GPU taskforce (M. Rivera) | Step 1 |
| 3 | Instantiate CouplingManager with explicit strategy, lag iterations configurable, and diagnostics logging residuals.[Coupling_GPU_Specification.md:21][Coupling_GPU_Specification.md:41][Coupling_GPU_Specification.md:43] | Explicit Solver team | Step 2 |
| 4 | Integrate InterfaceTracker updates for plate/fluid interface (mesh smoothing + ghost refresh) using Kokkos kernels.[Coupling_GPU_Specification.md:22][Coupling_GPU_Specification.md:37] | Mesh services (L. Patel) | Step 1 |
| 5 | Configure MPI domain decomposition and CUDA-aware exchanges for interface boundary layers; fall back to host staging to compare performance.[Coupling_GPU_Specification.md:37][Coupling_GPU_Specification.md:38] | Platform infra (A. Chen) | Steps 1-4 |
| 6 | Assemble benchmark case, run on workstation (1 GPU) and cluster (multi-GPU) capturing displacement, drag, and timing metrics.[Coupling_GPU_Specification.md:48][Coupling_GPU_Specification.md:59][Coupling_GPU_Specification.md:60] | Validation team | Steps 1-5 |

## 4. Milestones & Timeline

- **Week 1**: FieldRegistry + CouplingOperator stubs with CPU parity tests.
- **Week 2**: GPU kernels validated, CouplingManager diagnostics available.
- **Week 3**: InterfaceTracker and MPI pipeline integrated; first coupled timestep executed.
- **Week 4**: Benchmark runs completed with comparison plots vs. reference and OpenRadioss results.

## 5. Success Criteria

- Displacement and drag histories within 5% of reference solution and legacy OpenRadioss output.[Coupling_GPU_Specification.md:59]
- GPU execution achieves ≥2× speedup over CPU fallback for coupling operations (Step 2/3 kernels).[Coupling_GPU_Specification.md:30][Coupling_GPU_Specification.md:31]
- Diagnostics report stability warnings when CFL or coupling thresholds violated; no unexplained divergence.[Coupling_GPU_Specification.md:5][Coupling_GPU_Specification.md:41]

## 6. Deliverables

- Prototype configuration files, Catch2 unit tests, and benchmark scripts stored under `tests/benchmarks/fsi3`.
- Report summarising performance, accuracy, and lessons learned feeding into production coupling implementation.
