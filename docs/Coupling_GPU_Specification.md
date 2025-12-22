# Coupling & GPU Execution Specification
> See `README_NEXTGEN.md` for documentation map and reading order.

## 1. Coupling Framework Requirements

- Support explicit and implicit coupling strategies for FSI and FEM–meshfree interactions, with configurable stability controls and load mapping.[../../OpenRadioss/new_project_spec/specification.md:22][../../OpenRadioss/new_project_spec/specification.md:25][../../OpenRadioss/new_project_spec/specification.md:27]
- Manage data transfer across non-conforming meshes and particle systems, including dynamic interface tracking and adaptive partitioning.[../../OpenRadioss/new_project_spec/specification.md:26][../../OpenRadioss/new_project_spec/specification.md:33]
- Provide warning and diagnostic capabilities to surface interface instabilities and time-step constraints.[../../OpenRadioss/new_project_spec/specification.md:140]

## 2. Architecture Patterns (RadiossNX Reference)

- Implement observer-based physics modules to broadcast field updates, using field registries for type-safe data exchange.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:301][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:344]
- Apply strategy pattern for time integration, enabling explicit staggered and implicit monolithic solvers selectable at runtime.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:260][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:279]
- Use memory arenas and structure-of-arrays containers to expose coupled data sets efficiently to GPU kernels.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:215][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:246]

## 3. Coupling Components

| Component | Responsibility | Implementation Notes |
| --- | --- | --- |
| **FieldRegistry** | Register nodal/element/particle fields with metadata (location, stride, ownership). | Backed by Kokkos views; maintain host/device mirrors for CPU/GPU interoperability.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:246] |
| **CouplingOperator** | Transfer data between domains (e.g., structural displacement → fluid mesh motion). | Provide CPU and GPU implementations; support interpolation kernels leveraging Kokkos parallel_for. |
| **CouplingManager** | Schedule observer notifications, manage lag iterations for implicit coupling, monitor convergence. | Compose with strategy time stepper; integrate warning system for non-convergence or excessive corrections.[../../OpenRadioss/new_project_spec/specification.md:23][../../OpenRadioss/new_project_spec/specification.md:140] |
| **InterfaceTracker** | Handle interface topology updates (mesh adaptation, particle insertion/removal). | Use parallel graph updates and ghost layer refresh consistent with domain decomposition strategy.[../../OpenRadioss/new_project_spec/specification.md:27][../../OpenRadioss/new_project_spec/specification.md:167] |

## 4. GPU Kernel Plan

### 4.1 Target Kernels

- **Element Computations**: Port stress/strain evaluation and internal force assembly to Kokkos, aligning with existing dispatcher semantics (`forint.F`).[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:60][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:256]
- **Meshfree Interactions**: Implement SPH neighbour search, DEM contact, and blending functions using GPU-resident spatial hashing and reductions.[../../OpenRadioss/new_project_spec/specification.md:6][../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:78]
- **Coupling Transfers**: Offload interpolation matrices and load redistribution operations to GPUs to minimize communication overhead.[../../OpenRadioss/new_project_spec/specification.md:25][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:301]
- **Linear Algebra**: Bind PETSc/Trilinos GPU backends for implicit solves; supplement with custom CUDA/HIP kernels for block operations when required.[../../OpenRadioss/new_project_spec/specification.md:187][../../viRadioss/PROJECT_SUMMARY.md:76]

### 4.2 Execution Model

1. **Data Layout**: Store primary fields in device memory via Kokkos views; use `DualView` for host/device synchronization when CPU components (legacy Fortran) participate.
2. **Kernel Launch**: Each physics module offers GPU-ready entry points; fallback CPU implementations share interfaces for portability.
3. **MPI Integration**: Employ CUDA-aware MPI or HIP-aware MPI to exchange boundary data directly from device buffers; retain host staging path for compatibility.
4. **Task Scheduling**: Explore Kokkos task DAG or std::execution (when available) to overlap coupling operations with communication as suggested by RadiossNX task-based roadmap.[../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:19][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:33]

## 5. Stability & Diagnostics

- **Adaptive time stepping**: Integrate CFL-based step control with coupling-specific criteria, mirroring legacy behaviour while adding GPU-friendly reduction kernels for minimum time step evaluation.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:96][../../OpenRadioss/new_project_spec/specification.md:166]
- **Monitoring**: Provide per-iteration metrics (residual norms, interface penetration, load imbalance) accessible via logging and HDF5 diagnostics streams.[../../OpenRadioss/new_project_spec/specification.md:34][../../viRadioss/NEXTGEN_ARCHITECTURE_DESIGN.md:201]
- **Fallback Modes**: Allow CPU-only execution paths for modules not yet ported; automatically downgrade coupling operators when GPU resources unavailable.

## 6. Integration Workflow

1. **Define Fields**: Each physics module registers required fields (e.g., `solid.displacement`, `fluid.pressure`) with FieldRegistry.
2. **Bind Couplings**: Configuration maps source and target fields to CouplingOperator implementations (GPU or CPU).
3. **Time Step Loop**:
   - Advance physics modules via time-step strategy.
   - Execute coupling exchanges (GPU kernels) and enforce convergence criteria.
   - Synchronize MPI ghost layers using device-aware communication.
4. **Diagnostics**: Collect warnings and metrics; trigger fallback or error handling when thresholds exceeded.

## 7. Validation Path

- **Unit Tests**: Use Catch2 to validate coupling operators against analytical transfer scenarios; ensure GPU/CPU parity.
- **Integration Tests**: Run canonical FSI and SPH benchmark decks, comparing results against OpenRadioss outputs within tolerance.[../../viRadioss/OpenRadioss_Comprehensive_Architecture_Report.md:186]
- **Performance Benchmarks**: Measure coupling kernel speedups and MPI overlap efficiency on representative hardware (workstation + cluster).
