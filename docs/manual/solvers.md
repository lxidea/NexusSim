(part5)=
# Part V — Solvers

This part describes the explicit dynamics solver, the implicit solver framework, and the
nonlinear solution methods available in NexusSim.

---

(ch13_explicit)=
## Explicit Dynamics

The explicit dynamics solver (`fem/fem_solver.hpp`, `physics/time_integration.hpp`,
`physics/adaptive_timestep.hpp`) implements the central-difference time integration
scheme, which is conditionally stable and well-suited for short-duration, high-deformation
events such as impact, blast, and crash simulations.

### Central Difference Algorithm

Given the mass matrix $\mathbf{M}$, displacements $\mathbf{u}_n$, velocities
$\mathbf{v}_n$, accelerations $\mathbf{a}_n$, and time step $\Delta t$, the algorithm
proceeds as follows:

1. Compute accelerations:
$$
\mathbf{a}_n = \mathbf{M}^{-1}
\left(\mathbf{F}_{\text{ext}} - \mathbf{F}_{\text{int}}
- \mathbf{F}_{\text{contact}} - \mathbf{C}\mathbf{v}_n\right)
$$

2. Update velocities (half-step):
$$
\mathbf{v}_{n+1/2} = \mathbf{v}_{n-1/2} + \Delta t \, \mathbf{a}_n
$$

3. Update displacements:
$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t \, \mathbf{v}_{n+1/2}
$$

4. Compute internal forces from the updated configuration: $\mathbf{F}_{\text{int}}(\mathbf{u}_{n+1})$.
5. Apply contact forces, erosion criteria, and boundary conditions.

The use of a lumped (diagonal) mass matrix $\mathbf{M}$ makes the inversion trivial and
avoids the need to solve a linear system at each time step.

### Adaptive Time Step

The stable time step is determined by the CFL (Courant–Friedrichs–Lewy) condition:

$$
\Delta t = f_s \cdot \min_e \frac{L_e}{c_e}
$$

where $L_e$ is the characteristic length of element $e$, $c_e = \sqrt{E/\rho}$ is the
wave speed, and $f_s = 0.9$ is the safety factor.

```cpp
struct AdaptiveTimestep {
    Real compute_stable_dt(const Mesh& mesh, const State& state);
};
```

### Subcycling

The `SubcyclingController` class allows different element groups to be integrated at
different time steps. Small or stiff elements that impose a restrictive time step are
subcycled at a smaller $\Delta t$, while the remainder of the mesh uses the global
$\Delta t$.

```cpp
class SubcyclingController {
    void add_group(const std::string& name, Real dt_factor);
    int subcycle_ratio(const std::string& group);
};
```

### Energy Monitor

The `EnergyMonitor` class tracks energy balance throughout the simulation:

```cpp
class EnergyMonitor {
    Real kinetic_energy();
    Real internal_energy();
    Real contact_energy();
    Real hourglass_energy();
    Real total_energy();
    Real energy_error();    // (E_total − E_initial) / E_initial
};
```

Energy balance is a primary indicator of simulation quality. A well-conditioned explicit
simulation should maintain energy error below 5–10%.

### Element Erosion

The `ElementErosion` class (`physics/element_erosion.hpp`) provides over 15 failure
criteria for removing severely distorted elements from the simulation:

```cpp
class ElementErosion {
    void check_erosion(State& state, Real dt);
    void erode_element(Index elem_id);
};
```

**Available criteria:** Maximum principal strain, von Mises stress, hydrostatic pressure,
time step ratio, volume change, NaN detection, and user-defined criteria.

When an element is eroded, its mass is redistributed to its nodes and it ceases to
contribute internal forces.

### Wave 9: Explicit Solver Enhancements

The explicit solver was completed in Wave 9 (`fem/explicit_dynamics.hpp`) with the following additions:

**Bulk Viscosity** — Linear and quadratic artificial viscosity for shock capturing:

$$
q = C_1 \rho c L_e |\dot{\varepsilon}_v| + C_2 \rho L_e^2 \dot{\varepsilon}_v^2
$$

where $C_1$ and $C_2$ are the linear and quadratic viscosity coefficients.

**Hourglass Control** — Flanagan-Belytschko viscous and perturbation-based stiffness hourglass stabilization for reduced-integration elements.

**Energy Monitoring** — Per-step tracking of kinetic, internal, contact, hourglass, and external work energies with energy balance error checking.

**Element Erosion** — Stress/strain-based element deletion with mass conservation and neighbor contact update.

---

(ch14_implicit)=
## Implicit Solvers

The implicit solver framework (`solver/` directory) provides the infrastructure for
static and dynamic analyses that require the solution of large sparse linear systems.

### Sparse Matrix (CSR Format)

The `SparseMatrixCSR` class (`solver/sparse_matrix.hpp`) stores the global stiffness
matrix in Compressed Sparse Row (CSR) format:

```cpp
class SparseMatrixCSR {
public:
    SparseMatrixCSR(Index rows, Index cols);
    void build_from_coo(rows, cols, values);
    void add_value(Index row, Index col, Real value);
    void multiply(const std::vector<Real>& x, std::vector<Real>& y) const;
    Real diagonal(Index i) const;
    void scale(Real alpha);
    const std::vector<Index>& row_ptr() const;
    const std::vector<Index>& col_indices() const;
    std::vector<Real>& values();
};
```

A GPU-compatible sparse matrix class is also available (`solver/gpu_sparse_matrix.hpp`)
for systems compiled with Kokkos GPU backends.

### Linear Solvers

#### Preconditioned Conjugate Gradient

The `ConjugateGradientSolver` class implements the preconditioned conjugate gradient (PCG)
method for symmetric positive-definite systems:

```cpp
class ConjugateGradientSolver {
    struct Config {
        int max_iterations = 1000;
        Real tolerance = 1.0e-10;
        bool verbose = false;
    };

    SolverResult solve(const SparseMatrixCSR& A,
                       const std::vector<Real>& b,
                       std::vector<Real>& x,
                       const Preconditioner* precond = nullptr);
};
```

**Available preconditioners:**

| Preconditioner           | Description                          |
|--------------------------|--------------------------------------|
| `JacobiPreconditioner`   | Diagonal scaling: $M = \text{diag}(A)$ |
| `SSORPreconditioner`     | Symmetric SOR with relaxation parameter $\omega$ |

#### Direct Solver

The `DirectSolver` class performs LU factorization for small-to-medium systems:

```cpp
class DirectSolver {
    SolverResult solve(const SparseMatrixCSR& A,
                       const std::vector<Real>& b,
                       std::vector<Real>& x);
};
```

### Newton–Raphson Solver

The `NewtonRaphsonSolver` class (`solver/implicit_solver.hpp`) solves nonlinear systems
using Newton's method with optional line search:

```cpp
class NewtonRaphsonSolver {
    using InternalForceFunction =
        std::function<std::vector<Real>(const std::vector<Real>& u)>;
    using TangentFunction =
        std::function<SparseMatrixCSR(const std::vector<Real>& u)>;

    struct Config {
        int max_iterations = 20;
        Real tolerance = 1.0e-8;
        Real line_search_alpha = 1.0;
        bool use_line_search = true;
    };

    NewtonResult solve(const std::vector<Real>& F_ext,
                       std::vector<Real>& u,
                       InternalForceFunction F_int,
                       TangentFunction K_tangent);
};
```

The internal force and tangent stiffness are provided as callbacks, decoupling the solver
from the element-level computations.

### FEM Static Solver

The `FEMStaticSolver` class (`solver/fem_static_solver.hpp`) provides a high-level
interface for linear and nonlinear static analyses:

```cpp
class FEMStaticSolver {
public:
    FEMStaticSolver(Mesh& mesh);

    // Material assignment
    void set_material(Real E, Real nu);
    void set_shell_thickness(Index elem_id, Real thickness);

    // Boundary conditions
    void fix_node(Index node_id, int dof, Real value = 0.0);
    void fix_node_all(Index node_id);
    void add_force(Index node_id, int dof, Real value);
    void add_moment(Index node_id, int dof, Real value);

    // Solution
    void solve_linear();
    ArcLengthResult solve_arc_length(ArcLengthConfig config);

    // Post-processing
    std::vector<Real> get_node_displacement(Index node_id, int dpn = 3);
    Real max_displacement(int dpn = 3);
    int dof_per_node();
};
```

**Element dispatch:** The solver automatically detects element types in the mesh and
dispatches to the corresponding stiffness matrix routine. Supported element types are
Hex8, Hex20, Tet4, Tet10, and Shell4.

**Shell4 6-DOF auto-detection:** When shell elements are present in the mesh, the solver
automatically switches to 6 DOFs per node. Solid-only nodes in mixed meshes receive a
rotational penalty stiffness to prevent singularity.

### FEM Implicit Dynamic Solver

The `FEMImplicitDynamicSolver` class solves the transient equation of motion using the
Newmark-$\beta$ time integration scheme:

$$
\mathbf{M}\ddot{\mathbf{u}}_{n+1} + \mathbf{C}\dot{\mathbf{u}}_{n+1}
+ \mathbf{K}\mathbf{u}_{n+1} = \mathbf{F}_{n+1}
$$

**Newmark-$\beta$ update equations:**

$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t \, \dot{\mathbf{u}}_n
+ \Delta t^2 \left[(0.5 - \beta)\ddot{\mathbf{u}}_n
+ \beta \, \ddot{\mathbf{u}}_{n+1}\right]
$$

$$
\dot{\mathbf{u}}_{n+1} = \dot{\mathbf{u}}_n
+ \Delta t \left[(1 - \gamma)\ddot{\mathbf{u}}_n
+ \gamma \, \ddot{\mathbf{u}}_{n+1}\right]
$$

**Standard parameters:** $\beta = 0.25$ (average acceleration, unconditionally stable),
$\gamma = 0.5$ (no numerical damping).

**Effective stiffness:** $\mathbf{K}_{\text{eff}} = \mathbf{K} + a_0\mathbf{M} + a_1\mathbf{C}$

### Robustness Guards

The implicit solver includes comprehensive diagnostic checks:

- **NaN/Inf detection** — Applied to the right-hand-side vector, residual, $\mathbf{p}^T\mathbf{A}\mathbf{p}$ dot product, and solution vector.
- **Zero diagonal detection** — Identifies unconstrained degrees of freedom.
- **Degenerate element skip** — Elements with negative or zero Jacobian determinant are skipped during assembly.
- **Condition number estimate** — A warning is issued for ill-conditioned systems.
- **Solver diagnostics** — Detailed diagnostic messages are generated upon convergence failure.

---

(ch15_nonlinear)=
## Nonlinear Solution Methods

### Arc-Length Method (Crisfield's Cylindrical)

The arc-length solver (`solver/arc_length_solver.hpp`) traces nonlinear equilibrium paths,
including snap-through and snap-back behavior, by treating the load factor $\lambda$ as
an additional unknown.

**Constraint equation (cylindrical, $\psi = 0$):**

$$
\|\Delta\mathbf{u}\|^2 + \psi^2 (\Delta\lambda)^2 \|\mathbf{F}_{\text{ref}}\|^2
= \Delta\ell^2
$$

**Configuration:**

```cpp
struct ArcLengthConfig {
    Real arc_length = 1.0;
    Real psi = 0.0;                // 0 = cylindrical constraint
    int max_steps = 100;
    Real tolerance = 1.0e-6;
    int max_corrections = 15;
    Real lambda_max = 2.0;
    int desired_iterations = 5;    // For adaptive step sizing
    Real min_arc_length, max_arc_length;
    bool verbose = false;
};
```

**Algorithm (per step):**

1. **Predictor:** Solve $\mathbf{K}_t \, \delta\mathbf{u}_t = \mathbf{F}_{\text{ref}}$;
   compute $\Delta\lambda = \pm\Delta\ell \, / \, \|\delta\mathbf{u}_t\|$.

2. **Corrector (bordering):** Solve $\mathbf{K}_t \, \delta\mathbf{u}_R = -\mathbf{R}$
   and $\mathbf{K}_t \, \delta\mathbf{u}_F = \mathbf{F}_{\text{ref}}$; determine
   $\mathrm{d}\lambda$ from the quadratic constraint equation.

3. **Convergence check:** $\|\mathbf{R}\| \, / \, \|\mathbf{F}_{\text{ref}}\| < \text{tolerance}$.

4. **Adaptive step:** $\Delta\ell_{\text{new}} = \Delta\ell \sqrt{n_{\text{desired}} \, / \, n_{\text{actual}}}$, clamped to $[\Delta\ell_{\min}, \Delta\ell_{\max}]$.

**Quadratic for $\mathrm{d}\lambda$:**

$$
a \, \mathrm{d}\lambda^2 + b \, \mathrm{d}\lambda + c = 0
$$

where:

$$
a = \delta\mathbf{u}_F \cdot \delta\mathbf{u}_F + \psi^2 \|\mathbf{F}_{\text{ref}}\|^2
$$

$$
b = 2(\Delta\mathbf{u} + \delta\mathbf{u}_R) \cdot \delta\mathbf{u}_F
+ 2\psi^2 \Delta\lambda \|\mathbf{F}_{\text{ref}}\|^2
$$

$$
c = (\Delta\mathbf{u} + \delta\mathbf{u}_R) \cdot (\Delta\mathbf{u} + \delta\mathbf{u}_R)
+ \psi^2 \Delta\lambda^2 \|\mathbf{F}_{\text{ref}}\|^2 - \Delta\ell^2
$$

**Result structure:**

```cpp
struct ArcLengthResult {
    bool converged;
    int total_steps, total_iterations;
    Real final_load_factor;
    std::vector<PathPoint> path;   // (lambda, displacement, iterations) per step
};
```

### PETSc Integration

When PETSc is available (`NEXUSSIM_HAVE_PETSC`), the `PETScLinearSolver` class
(`solver/petsc_solver.hpp`) provides access to PETSc's scalable linear solvers:

```cpp
class PETScLinearSolver : public LinearSolver {
    void set_ksp_type(PETScKSPType type);
    void set_pc_type(PETScPCType type);
    SolverResult solve(const SparseMatrixCSR& A,
                       const std::vector<Real>& b,
                       std::vector<Real>& x);
};
```

**Supported Krylov subspace (KSP) types:** CG, GMRES, BiCGSTAB, PREONLY.

**Supported preconditioner (PC) types:** None, Jacobi, ILU, ICC, AMG, LU, Cholesky.

The solver converts NexusSim's CSR format to PETSc `Mat` and `Vec` objects internally
and calls `KSPSolve`. PETSc support is enabled via `cmake -DNEXUSSIM_ENABLE_PETSC=ON`.
