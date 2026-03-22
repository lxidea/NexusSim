# OpenRadioss to NexusSim Migration Guide

This guide helps users migrate existing OpenRadioss models to the NexusSim
framework.

---

## Why Migrate

| Feature             | OpenRadioss (Fortran)          | NexusSim (C++20/Kokkos)         |
|---------------------|--------------------------------|---------------------------------|
| Language            | Fortran 90/95                  | C++20, header-only              |
| GPU acceleration    | Limited CUDA kernels           | Kokkos portable (OpenMP, CUDA)  |
| Build system        | Custom Makefiles               | CMake 3.20+                     |
| Material interface  | Fixed-layout arrays            | `MaterialProperties` struct     |
| Extensibility       | Modify source, recompile       | Inherit `Material`, plug in     |
| Testing             | Separate QA scripts            | Integrated CTest, ~5,570 tests  |
| Modern features     | --                             | Templates, RAII, constexpr      |
| API documentation   | Doxygen (partial)              | Doxygen + Sphinx manual         |

---

## Keyword Mapping

### Top-Level Cards

| OpenRadioss Keyword  | NexusSim Equivalent                | Notes                         |
|----------------------|------------------------------------|-------------------------------|
| `/MAT/LAW*`          | `physics::MaterialProperties`      | Populate struct, pass to ctor |
| `/PROP/SHELL`        | `io::PropertyReader::read_shell()` | Returns `ShellProperty`       |
| `/PROP/SOLID`        | `io::PropertyReader::read_solid()` | Returns `SolidProperty`       |
| `/PROP/BEAM`         | `io::PropertyReader::read_beam()`  | Returns `BeamProperty`        |
| `/NODE`              | `io::StarterNode` vector           | Or direct coordinate arrays   |
| `/SHELL`             | `io::StarterElement` with type     | Element connectivity          |
| `/BRICK`             | `io::StarterElement` with type     | 8-node hex connectivity       |
| `/BCS`               | Boundary condition arrays          | Fixed DOF flags               |
| `/GRAV`              | `fem::GravityLoad`                 | Body force vector             |
| `/CLOAD`             | `fem::ConcentratedLoad`            | Nodal force vector            |
| `/INTER/TYPE*`       | Contact classes per type           | See contact mapping below     |
| `/FAIL/*`            | Failure model classes              | Wave 2, 11, 19, 33           |
| `/EOS/*`             | EOS classes in `physics` namespace | Wave 5, 13, 43               |
| `/ANIM`              | `io::RadiossAnimWriter`            | Binary animation output       |
| `/TH/*`              | `io::ReactionForcesTH`             | Time history extraction       |
| `/DT/NODA/CST`       | `solver::ImplicitDtControl`        | Adaptive time stepping        |

### Run Control

| OpenRadioss          | NexusSim                           | Notes                         |
|----------------------|------------------------------------|-------------------------------|
| `/RUN/name/1`        | Build config + CMake               | Compile-time configuration    |
| `/STOP`              | End time in solver constructor     | `end_time` parameter          |
| `/DT`                | Time step in solver loop           | Manual or `ImplicitDtControl` |
| `/PRINT`             | `io::QAPrintWriter`                | QA print output               |
| `/REPORT`            | `io::ReportGenerator`              | Summary report                |

---

## Material LAW Mapping (Top 30)

| OpenRadioss LAW | NexusSim Class                      | Category           |
|-----------------|-------------------------------------|--------------------|
| LAW0            | `RigidMaterial` / `NullMaterial`    | Basic              |
| LAW1            | `ElasticMaterial`                   | Basic              |
| LAW2            | `ElasticPlasticFail` / `JohnsonCookMaterial` | Metals    |
| LAW4            | `ElasticShellMaterial`              | Shell metals       |
| LAW5            | `ExplosiveBurnMaterial`             | Explosive          |
| LAW6            | `LESFluidMaterial`                  | Fluid              |
| LAW10           | `DruckerPragerMaterial` / `DP3Material` | Soil          |
| LAW12           | `OrthotropicMaterial`               | Anisotropic        |
| LAW14           | `SoilCapMaterial` / `GranularMaterial` | Soil            |
| LAW17           | `SteinbergGuinanMaterial`           | High-rate metals   |
| LAW19           | `FabricMaterial` / `FabricNLMaterial` | Fabric           |
| LAW21           | `ExtendedSoilMaterial`              | Soil               |
| LAW23           | `ZerilliArmstrongMaterial`          | High-rate metals   |
| LAW24           | `ConcreteMaterial` / `CDPM2Concrete` | Concrete         |
| LAW25           | `CompositePlyMaterial`              | Composites         |
| LAW27           | `MohrCoulombMaterial` / `BrittleFractureMaterial` | Soil |
| LAW28           | `HoneycombMaterial`                 | Crush              |
| LAW29           | `UserDefinedMaterial`               | User-defined       |
| LAW32           | `HillMaterial` / `OrthotropicHill`  | Anisotropic yield  |
| LAW33           | `FoamMaterial` / `CrushableFoamMaterial` | Foam          |
| LAW34           | `ViscoelasticMaterial` / `PronyMaterial` | Viscoelastic  |
| LAW36           | `PiecewiseLinearMaterial`           | Tabulated metals   |
| LAW37           | `BarlatMaterial` / `Barlat2000Material` | Sheet metals  |
| LAW40           | `FrequencyViscoelasticMaterial`     | Viscoelastic       |
| LAW42           | `NeoHookeanMaterial` / `MooneyRivlinMaterial` / `OgdenMaterial` | Hyperelastic |
| LAW44           | `NonlinearElasticMaterial` / `CowperSymondsMaterial` | Rate-dependent |
| LAW46           | `ThermalEPMaterial` / `CreepMaterial` | Thermal          |
| LAW53           | `CompositeDamageMaterial`           | Composites         |
| LAW59           | `CohesiveMaterial` / `SpotWeldMaterial` | Cohesive      |
| LAW80           | `ThermalMetallurgyMaterial`         | Metallurgy         |
| LAW81           | `DPCapMaterial`                     | Soil               |

---

## Input Format Differences

### OpenRadioss D00 Format

```
/MAT/LAW2/1
steel
#              RHO_I
              7850.0
#                  E                  NU
          2.100E+11          3.000E-01
#             Sigma_y            E_tan
          2.500E+08          5.000E+08
```

### NexusSim C++ Equivalent

```cpp
#include <nexussim/physics/material.hpp>

using namespace nxs::physics;

MaterialProperties props;
props.density = 7850.0;
props.E = 2.1e11;
props.nu = 0.3;
props.yield_stress = 2.5e8;
props.hardening_modulus = 5.0e8;
props.compute_derived();

// For Johnson-Cook:
JohnsonCookMaterial mat(props);
MaterialState state{};
state.strain[0] = 0.001;
mat.compute_stress(state);
// state.stress[0] now contains sigma_xx
```

### Reading an OpenRadioss Model

```cpp
#include <nexussim/io/reader_wave22.hpp>

nxs::io::RadiossD00Reader reader;
reader.read("model_0000.rad");

auto& nodes = reader.nodes();
auto& elements = reader.elements();
auto& materials = reader.materials();
```

---

## Output Format Equivalents

| OpenRadioss Output   | NexusSim Writer                  | File Extension |
|----------------------|----------------------------------|----------------|
| ANIM files           | `io::RadiossAnimWriter`          | `.anim`        |
| Status file          | `io::StatusFileWriter`           | `.sta`         |
| Time history         | `io::ReactionForcesTH`           | `.th`          |
| H3D (HyperView)     | `io::H3DWriter` (Wave 20)       | `.h3d`         |
| D3PLOT (LS-DYNA)     | `io::D3PLOTWriter` (Wave 20)    | `.d3plot`      |
| EnSight Gold         | `io::EnSightGoldWriter` (Wave 20)| `.case`       |
| Dynain restart       | `io::DynainWriter`               | `.k`           |
| QA print             | `io::QAPrintWriter`              | `.qa`          |
| Summary report       | `io::ReportGenerator`            | `.txt`         |
| Section forces       | `io::SectionForceOutput` (Wave 40)| `.secf`       |

---

## Common Migration Patterns

### Pattern 1: Material Definition

**Fortran (OpenRadioss)**:
```fortran
! LAW2 elastic-plastic
CALL LAW2_INIT(MAT_ID, RHO, E, NU, SIGY, ET)
CALL LAW2_SIGMA(NEL, SIG, DEPS, EPSOLD, ...)
```

**C++ (NexusSim)**:
```cpp
auto props = make_props(E, nu, rho, sigy);
props.hardening_modulus = Et;
physics::JohnsonCookMaterial mat(props);

physics::MaterialState state{};
state.strain[0] = eps_xx;
mat.compute_stress(state);
double sigma_xx = state.stress[0];
```

### Pattern 2: Element Loop

**Fortran (OpenRadioss)**:
```fortran
DO NEL = 1, NUMEL
    CALL SIGEPS(NEL, SIG, EPS, MAT_PARAM, UVAR)
ENDDO
```

**C++ (NexusSim)**:
```cpp
Kokkos::parallel_for("stress_update", num_elements,
    KOKKOS_LAMBDA(int e) {
        MaterialState state{};
        // fill state.strain from element strains
        mat.compute_stress(state);
        // store state.stress back to element
    });
```

### Pattern 3: Contact Interface

**Fortran (OpenRadioss)**:
```fortran
CALL INTER_SORT(NSPMD, ...)
CALL INTER_FORCE(INTBUF, ...)
```

**C++ (NexusSim)**:
```cpp
solver::ImplicitContactK contact(3);
solver::ContactPair cp;
cp.node1 = master_node;
cp.node2 = slave_node;
cp.gap = computed_gap;
cp.normal[0] = nx; cp.normal[1] = ny; cp.normal[2] = nz;
auto triplets = contact.compute_contact_stiffness({cp}, nullptr, nullptr, kn);
```

### Pattern 4: Time Integration

**Fortran (OpenRadioss)**:
```fortran
CALL RESOL(DT, TEND, CYCLE, ...)
```

**C++ (NexusSim)**:
```cpp
solver::ImplicitDtControl dtctrl(dt_min, dt_max);
double dt = dt_init;
while (time < t_end) {
    // Newton solve...
    dt = dtctrl.update_dt(dt, newton_iters, converged, max_newton);
    time += dt;
}
```

### Pattern 5: Output Writing

**Fortran (OpenRadioss)**:
```fortran
CALL ANIM_WRITE(IANIM, ...)
```

**C++ (NexusSim)**:
```cpp
io::RadiossAnimWriter anim;
anim.open("output.anim");
anim.write_frame(time, node_data, elem_data);
anim.close();
```

---

## Known Limitations

1. **MPI**: NexusSim includes production MPI support (Wave 45) with serial
   fallbacks via `#ifdef NEXUSSIM_HAVE_MPI`. System MPICH is auto-detected by
   CMake. Build with `-DNEXUSSIM_ENABLE_MPI=ON` (default) for multi-node runs.
   Note: MPICH with UCX device may hang in WSL2 environments; use native Linux
   or OpenMPI for multi-rank execution.

2. **External libraries**: Some features benefit from optional dependencies:
   - HDF5 for large-scale I/O
   - Eigen3 for dense linear algebra (fallback implementations provided)
   - spdlog for structured logging
   - yaml-cpp for configuration files

3. **GPU memory**: Models exceeding GPU memory (e.g., GT 1030 with 2 GB) must
   use CPU-only Kokkos backends or multi-GPU setups.

4. **User materials**: The `UserDefinedMaterial` interface provides a callback
   mechanism but does not support Fortran user subroutines directly. A C-compatible
   wrapper is needed for legacy Fortran user materials.

5. **Shell formulations**: The Shell4 element with full integration exhibits mild
   shear locking (~1.5% of beam theory). Use reduced integration or MITC4 for
   thin-shell-dominated problems.

6. **Implicit solver size**: The header-only `MUMPSSolver` is limited to 10,000
   DOFs. For larger problems, link against external MUMPS or PARDISO.

7. **Input format coverage**: The Radioss D00 reader covers the most common
   keywords. Rarely used cards may require manual translation.

---

## Quick Start

```bash
# Build NexusSim (MPI auto-detected from system MPICH/OpenMPI)
cmake -S . -B build -DNEXUSSIM_BUILD_PYTHON=OFF
cmake --build build -j$(nproc)

# Run validation suite
cd build && ctest --output-on-failure

# Run MPI tests with 2 ranks
ctest -R wave45 --output-on-failure
```

For GPU builds with CUDA:

```bash
cmake -S . -B build \
    -DKokkos_DIR=/usr/local/lib/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/usr/local/bin/nvcc_wrapper
cmake --build build -j$(nproc)
```

For serial-only builds (no MPI):

```bash
cmake -S . -B build \
    -DNEXUSSIM_ENABLE_MPI=OFF \
    -DNEXUSSIM_BUILD_PYTHON=OFF
cmake --build build -j$(nproc)
```

### Pattern 6: MPI Parallel Execution

**Fortran (OpenRadioss)**:
```fortran
CALL SPMD_EXCH_A(IAD_ELEM, FR_ELEM, FINT, ...)
CALL SPMD_GLOB_MIN(DT, DT_GLOB)
```

**C++ (NexusSim)**:
```cpp
#include <nexussim/parallel/force_exchange_wave45.hpp>

parallel::ForceExchanger exchanger;
exchanger.setup(partition, 3);  // 3 DOFs per node

// Accumulate ghost→owner forces
exchanger.begin_accumulate(forces);
exchanger.finish_accumulate(forces);

// Scatter owner→ghost accelerations
exchanger.begin_scatter(accelerations);
exchanger.finish_scatter(accelerations);

// Global time step synchronization
parallel::DistributedTimeStep dt_sync;
double global_dt = dt_sync.compute_global_dt(local_dt);
```
