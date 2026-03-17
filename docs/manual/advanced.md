(part8)=
# Part VIII — Advanced Features

This part describes the ALE mesh management module, the sensor and control subsystems,
the advanced capabilities (Wave 16), and the parallel computing framework (Wave 17).

---

(ch23_ale)=
## ALE Mesh Management

The Arbitrary Lagrangian–Eulerian (ALE) module (`physics/ale_solver.hpp`) combines the
advantages of the Lagrangian and Eulerian descriptions of motion. In the Lagrangian
phase, the mesh deforms with the material; in the ALE phase, the mesh is smoothed to
maintain element quality, and the state variables are advected (remapped) to the new
mesh positions.

### ALE Process

At each time step, the ALE solver performs three operations in sequence:

1. **Lagrangian step** — The standard FEM explicit or implicit update is applied,
   deforming the mesh with the material.
2. **Mesh smoothing** — Node positions are relocated to improve element quality while
   maintaining the domain boundary.
3. **Advection** — State variables (density, velocity, energy) are remapped from the
   Lagrangian mesh to the smoothed mesh.

### Smoothing Methods

| Method        | Algorithm                                              |
|---------------|--------------------------------------------------------|
| Laplacian     | $\mathbf{x}_{\text{new}} = (1-w)\mathbf{x} + w \, \overline{\mathbf{x}}_{\text{neighbors}}$ |
| Weighted      | Distance-weighted Laplacian (closer neighbors have more influence) |
| Equipotential | Solves the Laplace equation on the mesh (falls back to Laplacian) |

The smoothing weight $w \in [0, 1]$ controls the strength of the smoothing. Boundary
nodes are held fixed unless boundary smoothing is explicitly enabled.

### Advection Methods

| Method     | Order | Properties                                 |
|------------|-------|--------------------------------------------|
| Donor Cell | 1st   | Upwind scheme; simple, robust, diffusive   |
| Van Leer   | 2nd   | Minmod limiter; less diffusive, TVD-stable |

### API Reference

```cpp
class ALESolver {
public:
    // Smoothing
    Real smooth(Real* coordinates);

    // Advection
    void advect_scalar(const Real* old_coords, const Real* new_coords,
                       Real* field, Real dt);
    void advect_vector(const Real* old_coords, const Real* new_coords,
                       Real* field, Real dt);

    // Combined ALE step
    Real ale_step(Real* coords, Real* velocities,
                  const std::vector<Real*>& scalars, Real dt);

    // Element quality metrics
    static Real element_quality(const Real* coords, const Index* nodes, int npe);
    Real average_quality(const Real* coords, const Index* conn,
                         std::size_t ne, int npe) const;
};
```

The `element_quality()` method returns a value in $[0, 1]$, where $1$ represents a
perfectly regular element (cube for hexahedra, equilateral tetrahedron for tets). The
`ale_step()` method returns the average element quality after smoothing.

---

(ch24_sensors)=
## Sensors and Controls

The sensor and control subsystem enables runtime monitoring of simulation quantities and
automated response to user-defined conditions.

### Sensor Types

Five sensor types are provided (`fem/sensor.hpp`):

| Sensor          | Measurement                         | Parameters           |
|-----------------|-------------------------------------|----------------------|
| Accelerometer   | Nodal acceleration ($x$, $y$, or $z$) | `node_id`, `component` |
| Velocity Gauge  | Nodal velocity                      | `node_id`, `component` |
| Distance Sensor | Distance between two nodes          | `node_a`, `node_b`    |
| Strain Gauge    | Element strain component            | `elem_id`, `component` |
| Force Sensor    | Reaction force at constrained node  | `node_id`, `dof`       |

### Sensor API

```cpp
class Sensor {
    void sample(const State& state, Real time);
    Real current_value() const;
    const std::vector<std::pair<Real,Real>>& history() const;

    // CFC filtering (SAE J211 standard)
    void enable_cfc_filter(int cfc_class);
    Real filtered_value() const;
};

class SensorManager {
    Index add_sensor(const Sensor& sensor);
    void sample_all(const State& state, Real time);
    void set_sampling_interval(Real dt);
};
```

### CFC Filtering

The Channel Frequency Class (CFC) filter implements the SAE J211 standard for filtering
crash test data. Supported classes:

| CFC Class | Cutoff Frequency | Application                |
|-----------|------------------|----------------------------|
| CFC 60    | 100 Hz           | Structural deformation     |
| CFC 180   | 300 Hz           | Vehicle accelerations      |
| CFC 600   | 1,000 Hz         | Head/chest accelerations   |
| CFC 1000  | 1,650 Hz         | Helmet/component response  |

### Control System

The `ControlAction` class (`fem/controls.hpp`) defines condition–action pairs that
execute when a sensor reading crosses a threshold:

```cpp
class ControlAction {
    void set_sensor(Index sensor_id);
    void set_threshold(Real value);
    void set_action(ActionType type);
    void set_mode(ControlMode mode);    // OneShot or Repeating
    bool check_and_execute(const SensorManager& sensors,
                           State& state, Real time);
};
```

### Action Types

| Action Type         | Description                              |
|---------------------|------------------------------------------|
| Terminate           | End the simulation                       |
| Activate Load       | Turn on a load                           |
| Deactivate Load     | Turn off a load                          |
| Activate Contact    | Enable a contact interface               |
| Deactivate Contact  | Disable a contact interface              |
| Activate Failure    | Enable failure checking                  |
| Deactivate Failure  | Disable failure checking                 |
| Change Timestep     | Modify the time step                     |
| Write Checkpoint    | Trigger a checkpoint write               |
| Write Animation     | Trigger an animation frame               |
| Custom Callback     | Execute a user-defined `std::function`   |

---

(ch25_advanced_capabilities)=
## Advanced Capabilities (Wave 16)

The advanced capabilities module (`physics/advanced_wave16.hpp`) provides eight
sub-modules for specialized simulation scenarios.

### Modal Analysis (Lanczos Eigensolver)

The `LanczosEigensolver` class solves the generalized eigenvalue problem
$\mathbf{K}\mathbf{x} = \lambda\mathbf{M}\mathbf{x}$ for the lowest $N$ eigenpairs
using inverse iteration with M-orthogonal deflation. A CG inner solver handles the
linear system at each iteration, falling back to direct Gaussian elimination for systems
with $n \le 64$. Shift-invert mode ($\sigma$-shift) is supported for extracting interior
eigenvalues. Natural frequencies are obtained as $f_i = \sqrt{\lambda_i} / (2\pi)$.

### XFEM

The Extended Finite Element Method module models cracks without remeshing. The
`LevelSetCrack` class represents the crack geometry using two signed-distance fields:
$\phi$ (normal to the crack surface) and $\psi$ (tangential, ahead of the crack tip).
The `XFEMEnrichment` class provides Heaviside enrichment $H(\phi)$ for fully cut elements
and four asymptotic branch functions $\sqrt{r}\{\sin\frac{\theta}{2},
\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\sin\theta,
\cos\frac{\theta}{2}\sin\theta\}$ for tip elements. Crack propagation uses the
maximum hoop stress criterion (Erdogan-Sih) with J-integral evaluation.

### CONWEP Blast Loading

The `CONWEPBlast` class implements the Friedlander waveform for blast pressure-time
history:

$$
P(t) = P_s \left(1 - \frac{t}{t_{\text{pos}}}\right)
\exp\!\left(-b \, \frac{t}{t_{\text{pos}}}\right)
$$

Peak overpressure and positive-phase duration are computed from Kingery-Bulmash empirical
fits using Hopkinson-Cranz scaling $Z = R / W^{1/3}$. Oblique reflection coefficients
account for angle of incidence.

### Airbag Simulation

The `AirbagModel` class simulates gas-inflated airbag deployment using the ideal gas law
with mass inflow, vent holes, and fabric porosity. Mass conservation tracks inflator
inflow, orifice-based vent discharge (subsonic and choked flow regimes), and Darcy-law
fabric leakage. The energy balance couples pressure work $P \, dV$, inflator enthalpy
input, and outflow losses to update gas temperature and pressure each time step.

### Seatbelt Dynamics

The `BeltElement` class models belt webbing as a 1D tension-only bar element with
nonlinear stiffness, rate-dependent damping, and slip-ring feed-through using Euler's
belt friction formula $T_{\text{tight}} / T_{\text{slack}} = e^{\mu\theta}$. The
`Retractor` class adds spool-out, deceleration-triggered locking, pyrotechnic
pretensioner (timed force retraction), and a force-limiting load limiter that caps chest
loading above a specified threshold.

### Advanced ALE

The `EulerianSolver` class solves the 1D Euler equations on a fixed Cartesian grid using
first-order upwind (donor-cell) advection in conservation form for density, momentum,
and total energy. The `CutCellMethod` class tracks multi-material volume fractions (up to
four materials) with PLIC interface reconstruction and small-cell stabilization. The
`TurbulenceModel` class implements the standard $k$-$\varepsilon$ model (Launder-Sharma
constants) with wall functions for the log-law region.

### Adaptive Mesh Refinement

The `AMRManager` class implements h-refinement driven by the Zienkiewicz-Zhu (ZZ)
superconvergent patch recovery error estimator. Element errors are computed by comparing
FE stresses against neighbor-averaged recovered stresses. Elements are marked for
refinement when their error exceeds a fraction of the maximum, and for coarsening when
below a lower threshold. Quadrilateral elements are subdivided into four children, with
hanging-node constraints ($u_{\text{hang}} = \frac{1}{2}(u_L + u_R)$) enforced at
non-conforming edges.

### Draping Analysis

The `DrapingAnalysis` class performs kinematic fiber draping of composite fabrics onto
curved mold surfaces using the fishnet algorithm (Mack-Taylor). Starting from a pin
point, geodesic paths along warp and weft directions are propagated across the surface
mesh. At each cell the shear angle $\gamma = \pi/2 - \alpha$ (deviation from right-angle
fiber intersection) is computed from edge vectors. Elements exceeding a user-specified
lock angle are flagged as manufacturing-infeasible.

---

(ch26_parallel)=
## Parallel Computing (Wave 17)

The parallel computing module (`parallel/mpi_wave17.hpp`) provides six MPI-based
components for distributed simulation. All MPI calls are guarded by
`#ifdef NEXUSSIM_HAVE_MPI` with serial fallbacks for single-rank execution.

### Distributed Stiffness Assembly

The `DistributedAssembler` class manages per-rank assembly of the global stiffness matrix
in CSR format. Each rank owns a contiguous range of global rows and accumulates element
contributions as triplets (COO format), which are converted to CSR via
`build_csr_from_triplets()` with in-row column sorting and duplicate merging. Ghost node
force contributions are buffered separately and exchanged with owning ranks.

### Ghost Node Communication

The `GhostExchanger` class handles asynchronous field synchronization across partition
boundaries. Communication patterns are set up once from ghost-owner maps, then reused
each step via `begin_exchange()` / `finish_exchange()` (non-blocking `MPI_Isend` /
`MPI_Irecv`). Multi-DOF fields are supported through a configurable `dofs_per_node`
parameter that packs and unpacks vector-valued data (displacements, velocities, etc.).

### Domain Decomposition

The `DomainDecomposer` class provides two partitioning strategies. Recursive Coordinate
Bisection (RCB) splits along the longest dimension at weighted-median split points,
recursing until the target number of partitions is reached. The weighted greedy method
assigns elements to partitions using a score that balances load (lowest current weight)
against communication (most neighbors already assigned). Partition quality is reported as
load imbalance ratio and edge-cut count.

### Parallel Contact Detection

The `ParallelContactDetector` class implements a two-phase contact search. In the broad
phase, per-rank AABBs for contact surfaces A and B are exchanged to identify overlapping
rank pairs. In the narrow phase, element-level AABB intersection tests (inflated by a
search tolerance) produce `ContactPair` records containing surface IDs, owning ranks,
estimated gap distance, and approximate contact point coordinates.

### Load Balancing

The `LoadBalancer` class monitors per-rank computational weights and generates migration
plans when the imbalance ratio (max/average weight) exceeds a configurable threshold. The
greedy diffusion algorithm moves elements from overloaded ranks to the most underloaded
rank, subject to the constraint that transfers must strictly improve balance. The
resulting `MigrationPlan` records each element move with source rank, destination rank,
and element weight.

### Scalability Benchmarking

The `ScalabilityBenchmark` class provides timing infrastructure for parallel performance
analysis. Named phases (assembly, communication, solve) are timed individually via
`start_phase()` / `stop_phase()`. After collecting results across multiple rank counts,
`compute_scaling_metrics()` derives speedup $S(n) = T(1)/T(n)$, parallel efficiency
$E(n) = S(n)/n$, and communication-to-computation ratio. A formatted report is generated
via `report()`.
