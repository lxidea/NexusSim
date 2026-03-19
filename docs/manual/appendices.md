# Appendices

(appendix_a)=
## Appendix A — Complete Header File Index

The following is a listing of all 145 public header files, organized by
directory. Waves 18–22 added 19 headers across the `physics/`, `physics/failure/`,
`sph/`, `discretization/`, `io/`, `fem/`, and `parallel/` directories.

### core/ (7 files)

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `core.hpp`         | Aggregate include for all core headers       |
| `types.hpp`        | Fundamental types, vectors, enumerations, constants |
| `exception.hpp`    | Exception hierarchy and assertion macros      |
| `gpu.hpp`          | Kokkos abstraction layer, parallel wrappers   |
| `logger.hpp`       | Logging singleton and macros                  |
| `memory.hpp`       | Memory arena, pool, aligned buffer, tracker   |
| `mpi.hpp`          | MPI manager singleton                         |

### data/ (4 files)

| File          | Purpose                                      |
|---------------|----------------------------------------------|
| `data.hpp`    | Aggregate include for all data headers       |
| `field.hpp`   | Generic SOA field container                  |
| `mesh.hpp`    | Mesh: nodes, element blocks, node/side sets  |
| `state.hpp`   | State, StateHistory                          |

### coupling/ (3 files)

| File                   | Purpose                                  |
|------------------------|------------------------------------------|
| `coupling.hpp`         | Aggregate include                        |
| `coupling_operator.hpp`| Coupling operator base and implementations|
| `field_registry.hpp`   | Shared field registry singleton          |

### discretization/ (13 files)

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `hex8.hpp`         | 8-node hexahedron element                    |
| `hex20.hpp`        | 20-node serendipity hexahedron               |
| `tet4.hpp`         | 4-node tetrahedron                           |
| `tet10.hpp`        | 10-node quadratic tetrahedron                |
| `wedge6.hpp`       | 6-node wedge/prism                           |
| `shell3.hpp`       | 3-node triangular shell                      |
| `shell4.hpp`       | 4-node quadrilateral shell (6-DOF)           |
| `beam2.hpp`        | 2-node beam element                          |
| `truss.hpp`        | 2-node truss element                         |
| `spring_damper.hpp`| Spring, damper, spring-damper elements        |
| `mesh_partition.hpp`| Mesh partitioning for MPI                   |
| `elements_wave15.hpp` | Wave 15 elements (thick shell, DKT/DKQ, etc.) |
| `elements_wave20.hpp` | Wave 20 elements (BT shell, MITC4, EAS, etc.) |

### solver/ (6 files)

| File                    | Purpose                                 |
|-------------------------|-----------------------------------------|
| `implicit_solver.hpp`   | Newton–Raphson solver                   |
| `fem_static_solver.hpp` | FEM static solver (linear + arc-length) |
| `arc_length_solver.hpp` | Crisfield's cylindrical arc-length      |
| `petsc_solver.hpp`      | PETSc linear solver integration         |
| `sparse_matrix.hpp`     | CSR sparse matrix                       |
| `gpu_sparse_matrix.hpp` | GPU-compatible sparse matrix            |

### physics/ (29 files)

| File                            | Purpose                              |
|---------------------------------|--------------------------------------|
| `element.hpp`                   | Element base class and factory       |
| `element_erosion.hpp`           | Element erosion criteria             |
| `material.hpp`                  | MaterialState, MaterialProperties, MaterialType enum |
| `material_models.hpp`           | Aggregate: elastic, Von Mises, Johnson-Cook, Neo-Hookean |
| `material_library.hpp`          | Predefined material presets          |
| `material_orthotropic.hpp`      | Orthotropic elastic                  |
| `material_mooney_rivlin.hpp`    | Mooney-Rivlin hyperelastic           |
| `material_ogden.hpp`            | Ogden hyperelastic                   |
| `material_piecewise_linear.hpp` | Piecewise-linear plasticity          |
| `material_tabulated.hpp`        | Tabulated stress-strain              |
| `material_foam.hpp`             | Low-density foam                     |
| `material_crushable_foam.hpp`   | Crushable foam                       |
| `material_honeycomb.hpp`        | Honeycomb orthotropic crush          |
| `material_viscoelastic.hpp`     | Viscoelastic (Prony series)          |
| `material_cowper_symonds.hpp`   | Cowper-Symonds rate-dependent         |
| `material_zhao.hpp`             | Zhao rate-dependent                  |
| `material_elastic_plastic_fail.hpp` | Elastic-plastic with failure     |
| `material_rigid.hpp`            | Rigid material                       |
| `material_null.hpp`             | Null (mass-only) material            |
| `eos.hpp`                       | Equations of state (5 models)        |
| `section.hpp`                   | Section properties                   |
| `large_deformation.hpp`         | Large deformation utilities          |
| `time_integration.hpp`          | Time integration schemes             |
| `time_integrator.hpp`           | Time integrator interface            |
| `adaptive_timestep.hpp`         | Adaptive time step controller        |
| `ale_solver.hpp`                | ALE mesh management                  |
| `thermal_solver.hpp`            | Thermal/thermo-mechanical solver     |
| `module.hpp`                    | Physics module interface             |
| `physics.hpp`                   | Aggregate include                    |

### physics/ (continued — Waves 9–18)

| File                            | Purpose                              |
|---------------------------------|--------------------------------------|
| `explicit_dynamics.hpp`         | Explicit solver enhancements (Wave 9)|
| `material_wave10.hpp`           | 20 Wave 10 material models           |
| `eos_wave13.hpp`                | 8 Wave 13 EOS models                 |
| `thermal_wave14.hpp`            | Thermal solver (Wave 14)             |
| `elements_wave15.hpp`           | 7 element formulations (Wave 15)     |
| `advanced_wave16.hpp`           | 8 advanced capabilities (Wave 16)    |
| `material_wave18.hpp`           | 20 Tier 2 material models (Wave 18)  |

### physics/failure/ (8 + 2 files)

| File                    | Purpose                                    |
|-------------------------|--------------------------------------------|
| `failure_model.hpp`     | FailureState, FailureModel base class      |
| `failure_models.hpp`    | Aggregate include for all failure models   |
| `failure_hashin.hpp`    | Hashin 4-mode composite failure            |
| `failure_tsai_wu.hpp`   | Tsai-Wu quadratic interaction              |
| `failure_chang_chang.hpp`| Chang-Chang modified Hashin               |
| `failure_gtn.hpp`       | GTN ductile void growth                    |
| `failure_gissmo.hpp`    | GISSMO incremental damage                  |
| `failure_tabulated.hpp` | Tabulated failure strain                   |
| `failure_wave11.hpp`    | 12 Wave 11 failure models                  |
| `failure_wave19.hpp`    | 10 Wave 19 failure models                  |

### physics/composite/ (5 files)

| File                             | Purpose                             |
|----------------------------------|-------------------------------------|
| `composite_layup.hpp`            | Ply definition, CompositeLaminate, ABD matrix |
| `composite_thermal.hpp`          | Thermal residual stress             |
| `composite_interlaminar.hpp`     | Interlaminar shear stress           |
| `composite_progressive_failure.hpp` | FPF, ply degradation, strength envelope |
| `composite_utils.hpp`            | Laminate utilities                  |

### fem/ (22 files)

| File                       | Purpose                                |
|----------------------------|----------------------------------------|
| `fem_solver.hpp`           | Explicit FEM solver                    |
| `contact.hpp`              | Penalty contact                        |
| `edge_contact.hpp`         | Edge-based contact                     |
| `surface_contact.hpp`      | Surface-to-surface contact             |
| `node_to_surface_contact.hpp` | Node-to-surface projection          |
| `hertzian_contact.hpp`     | Hertzian sphere contact                |
| `mortar_contact.hpp`       | Mortar segment-to-segment              |
| `tied_contact.hpp`         | Tied contact with failure              |
| `voxel_collision.hpp`      | Voxel-based collision detection        |
| `friction_model.hpp`       | Coulomb friction with decay            |
| `rigid_body.hpp`           | Rigid body dynamics                    |
| `rigid_wall.hpp`           | Rigid wall constraints                 |
| `constraints.hpp`          | RBE2, RBE3, joints                     |
| `load_curve.hpp`           | Load curve and manager                 |
| `loads.hpp`                | Load manager                           |
| `initial_conditions.hpp`   | Initial velocity, temperature          |
| `sensor.hpp`               | Sensor types and manager               |
| `controls.hpp`             | Control actions and logic              |
| `contact_wave12.hpp`       | 9 Wave 12 contact types                |
| `contact_wave21.hpp`       | 6 Wave 21 contact refinements          |
| `ale_wave21.hpp`           | 6 Wave 21 ALE features                 |
| `solver_wave22.hpp`        | 5 Wave 22 solver hardening features    |

### io/ (18 files)

| File                          | Purpose                              |
|-------------------------------|--------------------------------------|
| `mesh_reader.hpp`             | Generic mesh reader interface        |
| `mesh_validator.hpp`          | Mesh quality validation              |
| `lsdyna_reader.hpp`           | LS-DYNA keyword file reader          |
| `radioss_reader.hpp`          | Radioss Starter deck reader          |
| `config_reader.hpp`           | YAML configuration parser            |
| `vtk_writer.hpp`              | VTK output (legacy + XML)            |
| `checkpoint.hpp`              | Basic binary checkpoint              |
| `extended_checkpoint.hpp`     | Extended checkpoint (history, contact)|
| `time_history.hpp`            | Time history output                  |
| `animation_writer.hpp`        | Animation sequence writer            |
| `part_energy.hpp`             | Part energy output                   |
| `interface_force_output.hpp`  | Interface force output               |
| `cross_section_force.hpp`     | Cross-section force/moment output    |
| `result_database.hpp`         | Binary result database               |
| `output_wave20.hpp`           | 6 production output formats (Wave 20)|
| `reader_wave22.hpp`           | 4 input readers (Wave 22)            |
| `preprocess_wave22.hpp`       | 5 preprocessing utilities (Wave 22)  |

### sph/ (5 files)

| File                  | Purpose                                   |
|-----------------------|-------------------------------------------|
| `sph_kernel.hpp`      | SPH kernel functions                      |
| `sph_solver.hpp`      | WCSPH solver                              |
| `neighbor_search.hpp` | Spatial hashing neighbor search           |
| `fem_sph_coupling.hpp`| FEM-SPH coupling                          |
| `sph_wave19.hpp`      | 7 SPH enrichment features (Wave 19)       |

### parallel/ (2 files)

| File                  | Purpose                                   |
|-----------------------|-------------------------------------------|
| `mpi_wave17.hpp`      | 6 MPI components (Wave 17)                |

### peridynamics/ (15 files)

| File                       | Purpose                                |
|----------------------------|----------------------------------------|
| `pd_types.hpp`             | PD Kokkos view type aliases            |
| `pd_particle.hpp`          | PDParticles data structure             |
| `pd_neighbor.hpp`          | PDNeighborList (CSR format)            |
| `pd_force.hpp`             | Bond-based force computation           |
| `pd_solver.hpp`            | PD time integration solver             |
| `pd_state_based.hpp`       | Ordinary state-based PD               |
| `pd_correspondence.hpp`    | Non-ordinary correspondence PD         |
| `pd_bond_models.hpp`       | Enhanced bond models                   |
| `pd_materials.hpp`         | JC, Drucker-Prager, JH2 materials     |
| `pd_morphing.hpp`          | FEM-to-PD element morphing             |
| `pd_mortar_coupling.hpp`   | Mortar FEM-PD coupling                 |
| `pd_adaptive_coupling.hpp` | Damage-driven adaptive coupling        |
| `pd_fem_coupling.hpp`      | General FEM-PD coupling interface      |
| `pd_contact.hpp`           | PD-based contact                       |
| `peridynamics.hpp`         | Aggregate include                      |

### utils/ (1 file)

| File                    | Purpose                                  |
|-------------------------|------------------------------------------|
| `performance_timer.hpp` | Timer, ScopedTimer, Profiler, BenchmarkResult |

---

(appendix_b)=
## Appendix B — Mathematical Notation

### Tensor Notation

| Symbol                              | Description                                |
|-------------------------------------|--------------------------------------------|
| $\boldsymbol{\sigma}$               | Cauchy stress tensor                       |
| $\boldsymbol{\varepsilon}$          | Infinitesimal strain tensor                |
| $\mathbf{F}$                        | Deformation gradient: $F_{ij} = \partial x_i / \partial X_j$ |
| $\mathbf{B}$                        | Left Cauchy–Green: $\mathbf{B} = \mathbf{F}\mathbf{F}^T$ |
| $\mathbf{C}$                        | Right Cauchy–Green: $\mathbf{C} = \mathbf{F}^T\mathbf{F}$ |
| $J$                                 | Jacobian determinant: $J = \det(\mathbf{F})$ |
| $\mathbf{I}$                        | Second-order identity tensor               |
| $\delta_{ij}$                       | Kronecker delta                            |
| $\epsilon_{ijk}$                    | Levi-Civita permutation symbol             |
| $\mathbf{D}$                        | Rate of deformation tensor                 |
| $\mathbf{W}$                        | Spin (vorticity) tensor                    |

### Voigt Convention

The Voigt notation maps symmetric tensors to vectors:

$$
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13} \\
\sigma_{12} & \sigma_{22} & \sigma_{23} \\
\sigma_{13} & \sigma_{23} & \sigma_{33}
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
\sigma_1 \\ \sigma_2 \\ \sigma_3 \\ \sigma_4 \\ \sigma_5 \\ \sigma_6
\end{bmatrix}
=
\begin{bmatrix}
\sigma_{xx} \\ \sigma_{yy} \\ \sigma_{zz} \\
\sigma_{xy} \\ \sigma_{yz} \\ \sigma_{xz}
\end{bmatrix}
$$

Engineering shear strains: $\gamma_{ij} = 2\varepsilon_{ij}$ for $i \neq j$.

### Continuum Mechanics

**Strong form** (balance of linear momentum):

$$
\rho \, \ddot{\mathbf{u}} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{b}
\quad \text{in } \Omega
$$

$$
\mathbf{u} = \bar{\mathbf{u}} \quad \text{on } \Gamma_D
\qquad
\boldsymbol{\sigma} \cdot \mathbf{n} = \bar{\mathbf{t}} \quad \text{on } \Gamma_N
$$

**Weak form** (principle of virtual work):

$$
\int_\Omega \boldsymbol{\sigma} : \delta\boldsymbol{\varepsilon} \, \mathrm{d}V
= \int_\Omega \mathbf{b} \cdot \delta\mathbf{u} \, \mathrm{d}V
+ \int_{\Gamma_N} \bar{\mathbf{t}} \cdot \delta\mathbf{u} \, \mathrm{d}A
$$

**Discretized form:**

$$
\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}}
+ \mathbf{F}_{\text{int}}(\mathbf{u}) = \mathbf{F}_{\text{ext}}
$$

### Lamé Parameters

$$
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad
\mu = G = \frac{E}{2(1+\nu)}
$$

### Bulk and Shear Moduli

$$
K = \frac{E}{3(1-2\nu)}, \qquad
G = \frac{E}{2(1+\nu)}
$$

---

(appendix_c)=
## Appendix C — Material Parameter Reference

### Elastic Materials

| Material                | $E$ (GPa) | $\nu$ | $\rho$ (kg/m³) | $\sigma_y$ (MPa) |
|-------------------------|-----------|--------|-----------------|-------------------|
| Mild Steel              | 210       | 0.30   | 7,850           | 250               |
| Stainless Steel 304     | 193       | 0.29   | 8,000           | 215               |
| Aluminum 6061-T6        | 69        | 0.33   | 2,700           | 276               |
| Aluminum 7075-T6        | 72        | 0.33   | 2,810           | 503               |
| Titanium Ti-6Al-4V      | 114       | 0.34   | 4,430           | 880               |
| Copper OFHC             | 117       | 0.34   | 8,960           | 33                |
| Concrete                | 30        | 0.20   | 2,400           | —                 |
| Glass (Soda-Lime)       | 72        | 0.22   | 2,500           | —                 |

### Hyperelastic Materials

| Material                | $C_{10}$ (MPa) | $C_{01}$ (MPa) | $D_1$ (1/MPa)  | $\rho$ (kg/m³) |
|-------------------------|----------------|-----------------|-----------------|-----------------|
| Natural Rubber          | 0.16           | 0.04            | 0.005           | 1,100           |
| Silicone Rubber         | 0.12           | 0.03            | 0.01            | 1,200           |
| Neoprene                | 0.30           | 0.08            | 0.003           | 1,230           |

### Foam Materials

| Material       | $E$ (MPa) | $\nu$ | $\rho$ (kg/m³) | $\sigma_y$ (MPa) | $\varepsilon_d$ |
|----------------|-----------|--------|-----------------|-------------------|-----------------|
| EPS Foam       | 5         | 0.10   | 30              | 0.1               | 0.85            |
| PU Foam        | 20        | 0.15   | 60              | 0.4               | 0.80            |
| Aluminum Foam  | 1,000     | 0.30   | 300             | 2.0               | 0.70            |

### Johnson-Cook Parameters

| Material       | $A$ (MPa) | $B$ (MPa) | $n$    | $C$    | $m$   |
|----------------|-----------|-----------|--------|--------|-------|
| Steel 4340     | 792       | 510       | 0.26   | 0.014  | 1.03  |
| Al 7075-T6     | 520       | 477       | 0.52   | 0.001  | 1.61  |
| Ti-6Al-4V      | 1,098     | 1,092     | 0.93   | 0.014  | 1.10  |
| OFHC Copper    | 90        | 292       | 0.31   | 0.025  | 1.09  |

---

(appendix_d)=
## Appendix D — Element Reference Cards

### Hex8 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 8-node hexahedron                             |
| Nodes                | 8 (corner)                                    |
| DOFs per node        | 3 (translation)                               |
| Total DOFs           | 24                                            |
| Shape functions      | Trilinear                                     |
| Integration          | $2 \times 2 \times 2$ = 8 Gauss points       |
| Topology             | Hexahedron                                    |
| Known issues         | Shear locking in bending                      |
| Recommended for      | Regular geometry, moderate deformation        |

### Hex20 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 20-node serendipity hexahedron                |
| Nodes                | 20 (8 corner + 12 mid-edge)                  |
| DOFs per node        | 3 (translation)                               |
| Total DOFs           | 60                                            |
| Shape functions      | Serendipity quadratic                         |
| Integration          | $3 \times 3 \times 3$ = 27 Gauss points      |
| Known issues         | High computational cost                       |
| Recommended for      | Stress concentrations, curved geometry        |

### Tet4 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 4-node tetrahedron                            |
| Nodes                | 4 (corner)                                    |
| DOFs per node        | 3 (translation)                               |
| Total DOFs           | 12                                            |
| Shape functions      | Linear (volume coordinates)                   |
| Integration          | 1 Gauss point (centroid)                      |
| Known issues         | Constant strain, volumetric locking           |
| Recommended for      | Complex geometry, automatic meshing           |

### Tet10 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 10-node quadratic tetrahedron                 |
| Nodes                | 10 (4 corner + 6 mid-edge)                   |
| DOFs per node        | 3 (translation)                               |
| Total DOFs           | 30                                            |
| Shape functions      | Quadratic                                     |
| Integration          | 4 Gauss points                                |
| Recommended for      | Accurate results with tet meshes              |

### Shell4 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 4-node quadrilateral shell                    |
| Nodes                | 4 (corner)                                    |
| DOFs per node        | 6 (3 translation + 3 rotation)                |
| Total DOFs           | 24                                            |
| Shape functions      | Bilinear                                      |
| Integration          | $2 \times 2$ = 4 Gauss points                |
| Theory               | Reissner–Mindlin (thick shell)                |
| Known issues         | Shear locking (~1.5% of beam theory)          |
| Recommended for      | Thin-walled structures, vehicle panels        |

### Beam2 Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 2-node beam                                   |
| Nodes                | 2                                             |
| DOFs per node        | 6 (3 translation + 3 rotation)                |
| Total DOFs           | 12                                            |
| Shape functions      | Linear (axial) / Hermite cubic (bending)      |
| Integration          | 2 Gauss points                                |
| Recommended for      | Frame structures, reinforcement               |

### Truss Reference Card

| Property             | Value                                         |
|----------------------|-----------------------------------------------|
| Type                 | 2-node truss                                  |
| Nodes                | 2                                             |
| DOFs per node        | 3 (translation)                               |
| Total DOFs           | 6                                             |
| Shape functions      | Linear                                        |
| Integration          | 1 Gauss point                                 |
| Recommended for      | Cable structures, bracing                     |
