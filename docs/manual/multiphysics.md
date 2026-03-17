(part7)=
# Part VII — Multi-Physics

This part describes the meshfree methods (SPH and Peridynamics), the multi-physics
coupling framework, and the thermal analysis module.

---

(ch19_sph)=
## Smoothed Particle Hydrodynamics

The SPH module (`sph/` directory) implements a weakly compressible SPH (WCSPH)
formulation for fluid and free-surface flow problems.

### SPH Kernels

The kernel function $W(r, h)$ is the fundamental building block of SPH interpolation.
Three kernels are provided:

| Kernel          | Continuity | Support | Properties                    |
|-----------------|------------|---------|-------------------------------|
| Cubic B-spline  | $C^2$      | $2h$    | Standard SPH kernel            |
| Wendland C4     | $C^4$      | $2h$    | Higher smoothness, less tensile instability |
| Quintic         | $C^4$      | $3h$    | Very smooth, larger support    |

```cpp
class SPHKernel {
    Real W(Real r, Real h) const;        // Kernel value
    Real dWdr(Real r, Real h) const;     // Kernel gradient
    Real support_radius(Real h) const;   // Compact support radius
};
```

### SPH Solver

The `SPHSolver` class implements the WCSPH algorithm:

```cpp
class SPHSolver {
    void compute_density();      // ρ_i = Σ_j m_j W(|x_i − x_j|, h)
    void compute_forces();       // Pressure + viscosity + body forces
    void time_integrate(Real dt);
};
```

**Density summation:**

$$
\rho_i = \sum_j m_j \, W(|\mathbf{x}_i - \mathbf{x}_j|, h)
$$

**Momentum equation:**

$$
\frac{\mathrm{d}\mathbf{v}_i}{\mathrm{d}t}
= -\sum_j m_j \left(\frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2}\right)
\nabla W_{ij} + \mathbf{g}
$$

**Equation of state (Tait):**

$$
P = \frac{c^2 \rho_0}{\gamma}\left[\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right]
\approx c^2 (\rho - \rho_0)
$$

### Neighbor Search

The `NeighborSearch` class (`sph/neighbor_search.hpp`) implements spatial hashing for
efficient neighbor finding within the kernel support radius.

### FEM-SPH Coupling

The `FEMSPHCoupling` class (`sph/fem_sph_coupling.hpp`) enables coupling between the FEM
domain and the SPH domain at their interface:

```cpp
class FEMSPHCoupling {
    void detect_interface(const Mesh& fem_mesh, const SPHParticles& sph);
    void transfer_pressure();      // SPH pressure → FEM surface loads
    void transfer_velocity();      // FEM velocity → SPH boundary particles
};
```

**Coupling methods:**

- **Penalty:** Spring forces at the interface proportional to penetration.
- **Pressure:** Direct pressure interpolation between the FEM surface and SPH particles.

---

(ch20_peridynamics)=
## Peridynamics

The peridynamics module (`peridynamics/` directory) provides a complete implementation
of bond-based, ordinary state-based, and non-ordinary correspondence peridynamics with
specialized constitutive models for metals, ceramics, and geomaterials.

### Data Structures

Peridynamics uses Kokkos views for GPU-portable data:

```cpp
using PDScalarView = Kokkos::View<Real*>;
using PDVectorView = Kokkos::View<Real*[3]>;
using PDIntView    = Kokkos::View<Index*>;
using PDBoolView   = Kokkos::View<bool*>;
```

**`PDParticles`** stores the particle state:

| View       | Description                    |
|------------|--------------------------------|
| `x()`      | Reference positions            |
| `u()`      | Displacements                  |
| `v()`      | Velocities                     |
| `f()`      | Forces                         |
| `volume()` | Particle volumes               |
| `density()`| Material densities             |
| `active()` | Active/inactive flags          |

**`PDNeighborList`** stores the bond connectivity in CSR format:

| View              | Description                        |
|-------------------|------------------------------------|
| `neighbor_offset()`| CSR-style row offsets             |
| `neighbor_list()` | Neighbor particle indices           |
| `neighbor_count()`| Number of neighbors per particle   |
| `bond_weight()`   | Influence function weights          |
| `bond_intact()`   | Bond integrity flags                |
| `bond_xi()`       | Reference bond vectors $\boldsymbol{\xi} = \mathbf{x}_j - \mathbf{x}_i$ |
| `bond_length()`   | Reference bond lengths $|\boldsymbol{\xi}|$ |

### Bond-Based Peridynamics (PMB Model)

The Prototype Microelastic Brittle (PMB) model computes bond forces as:

$$
\mathbf{f}(\boldsymbol{\xi}, \boldsymbol{\eta})
= c \, s \, w(|\boldsymbol{\xi}|) \, \mathbf{e}
$$

where:

- $\boldsymbol{\xi} = \mathbf{x}_j - \mathbf{x}_i$ is the reference bond vector.
- $\boldsymbol{\eta} = \mathbf{u}_j - \mathbf{u}_i$ is the relative displacement.
- $s = (|\boldsymbol{\xi} + \boldsymbol{\eta}| - |\boldsymbol{\xi}|) / |\boldsymbol{\xi}|$ is the bond stretch.
- $\mathbf{e} = (\boldsymbol{\xi} + \boldsymbol{\eta}) / |\boldsymbol{\xi} + \boldsymbol{\eta}|$ is the unit direction.
- $c = 18K / (\pi\delta^4)$ is the three-dimensional micro-modulus.
- $w$ is the influence function.

**Bond failure:** A bond breaks irreversibly when $s > s_c$, where:

$$
s_c = \sqrt{\frac{5 G_c}{9 K \delta}}
$$

and $G_c$ is the critical energy release rate.

### Ordinary State-Based Peridynamics

The ordinary state-based formulation separates volumetric and deviatoric deformation:

$$
\mathbf{f} = \frac{3K\theta}{m} w |\boldsymbol{\xi}| \, \mathbf{e}
+ \frac{15G}{m} w \left(|\boldsymbol{\eta} + \boldsymbol{\xi}|
- |\boldsymbol{\xi}|\left(1 + \frac{\theta}{3}\right)\right) \mathbf{e}
$$

where $\theta$ is the dilatation and $m$ is the weighted volume.

### Non-Ordinary Correspondence

The correspondence formulation computes the deformation gradient $\mathbf{F}$ from
peridynamic kinematics and applies standard continuum constitutive models:

$$
\mathbf{F} = \left[\sum_j (\boldsymbol{\eta}_j \otimes \boldsymbol{\xi}_j)
V_j w_j\right] \mathbf{K}^{-1}
$$

where $\mathbf{K} = \sum_j (\boldsymbol{\xi}_j \otimes \boldsymbol{\xi}_j) V_j w_j$ is
the shape tensor.

**Available constitutive models:** Neo-Hookean, linear elastic, Johnson-Cook.

**Hourglass control:** Penalty-based stabilization suppresses zero-energy modes inherent
in the correspondence formulation.

### Enhanced Bond Models

| Model               | Description                                     | History Variables        |
|---------------------|-------------------------------------------------|--------------------------|
| Energy-Based        | Bond breaks when accumulated strain energy > $G_c |\boldsymbol{\xi}|$ | `history[0]` = energy  |
| Microplastic        | Elasto-plastic bond with yield stretch           | `history[0]` = plastic stretch |
| Viscoelastic        | Standard linear solid with relaxation            | `history[1]` = viscous variable |
| Short-Range Repulsion | Compressive contact force only                 | None                     |

### PD Material Models

Three specialized PD material models are available:

**Johnson-Cook** — For metals at high strain rates, with damage parameters $D_1$–$D_5$.

**Drucker-Prager** — For geomaterials (soil, rock), parameterized by friction angle
$\phi$, cohesion $c$, and dilation angle $\psi$.

**Johnson-Holmquist 2 (JH2)** — For ceramics and glass, with separate intact and
fractured strength surfaces, progressive damage, and a polynomial EOS.

**Material presets:** Al7075-T6, Steel 4340, Ti-6Al-4V, Copper OFHC, LooseSand,
DenseSand, Clay, Concrete, Granite, Alumina, SiC, B4C, Soda-Lime Glass.

---

(ch21_coupling)=
## Multi-Physics Coupling

The coupling framework (`coupling/` directory) enables data transfer between different
physics modules. It is used for FEM–SPH and FEM–PD coupling.

### Field Registry

The `FieldRegistry` singleton (`coupling/field_registry.hpp`) manages shared fields
between physics modules:

```cpp
class FieldRegistry {
    static FieldRegistry& instance();
    void register_field(name, provider_module, field_ptr);
    void register_consumer(name, consumer_module);
    FieldPtr get_field(name);
    void mark_synchronized(name);
};
```

### Coupling Operators

The `CouplingOperator` base class (`coupling/coupling_operator.hpp`) defines the
interface for field transfer between discretizations:

```cpp
class CouplingOperator {
    virtual void initialize(const CouplingInterface& interface) = 0;
    virtual void transfer(const Field<Real>& source, Field<Real>& target) = 0;
};
```

| Operator              | Description                                     |
|-----------------------|-------------------------------------------------|
| `DirectCouplingOperator`         | 1:1 node mapping, same discretization |
| `InterpolationCouplingOperator`  | Weighted interpolation with neighbor weights |
| `NearestNeighborCouplingOperator`| Nearest node mapping                  |

The `InterpolationCouplingOperator` uses a GPU-compatible Kokkos kernel for parallel
interpolation:

```cpp
struct InterpolationKernel {
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const;
};
```

### FEM-PD Coupling Methods

Four coupling strategies are available for FEM–PD coupling:

#### Arlequin Method

A blending-zone approach where FEM and PD overlap:

$$
\alpha_{\text{FEM}}(\mathbf{x}) + \alpha_{\text{PD}}(\mathbf{x}) = 1
\quad \text{(partition of unity)}
$$

The total energy is decomposed as:

$$
E_{\text{total}} = \alpha_{\text{FEM}} E_{\text{FEM}}
+ \alpha_{\text{PD}} E_{\text{PD}} + E_{\text{coupling}}
$$

#### Mortar Coupling

Segment-to-horizon coupling at the FEM–PD interface using mortar integrals for
consistent force transfer.

#### Element Morphing

Dynamic FEM-to-PD conversion based on element damage:

1. When element damage exceeds a threshold, the element's nodes are converted to PD
   particles.
2. PD neighbor lists are established for the new particles.
3. State variables (stress, strain, history) are transferred from the FEM element to
   the PD particles.
4. The FEM element is deactivated.

#### Adaptive Coupling

Damage-driven zone reclassification: regions begin as FEM elements, automatically
convert to PD particles when damage initiates, with dynamic blending zone management.

---

(ch22_thermal)=
## Thermal Analysis

The thermal solver (`physics/thermal_solver.hpp`) implements heat conduction with
optional coupling to the mechanical solver for thermo-mechanical analysis.

**Governing equation:**

$$
\rho c_p \frac{\partial T}{\partial t}
= \nabla \cdot (k \nabla T) + Q
$$

where $\rho$ is the density, $c_p$ is the specific heat capacity, $k$ is the thermal
conductivity, $T$ is the temperature, and $Q$ is the volumetric heat source.

**Thermo-mechanical coupling:** Temperature changes generate thermal strains:

$$
\varepsilon_{ij}^{\text{th}} = \alpha \, \Delta T \, \delta_{ij}
$$

where $\alpha$ is the coefficient of thermal expansion. These thermal strains are
subtracted from the total strain before computing mechanical stresses.

### Heat Conduction Solver

The `HeatConductionSolver` class (`physics/thermal_wave14.hpp`) implements an explicit
forward-Euler scheme with a lumped capacity matrix for transient heat conduction. Nodal
temperatures are updated using a connectivity-based finite-difference Laplacian stencil
where each neighbor $j$ contributes a flux $k A_{ij} (T_j - T_i) / d_{ij}$ to the
lumped heat rate. The update formula per node is:

$$
T_i^{n+1} = T_i^n + \frac{\Delta t}{\rho \, c_p}
\left[\sum_j \frac{k \, A_{ij}}{d_{ij} \, V_i}(T_j - T_i) + Q_i\right]
$$

### Convection Boundary Conditions

The `ConvectionBC` class applies Newton's law of cooling on boundary faces. For each
boundary node $i$ with exposed area $A$, the convective heat loss is:

$$
Q_{\text{conv}} = -h \, A \, (T_i - T_{\text{amb}})
$$

where $h$ is the convective heat transfer coefficient (W/m$^2$-K) and $T_{\text{amb}}$
is the ambient fluid temperature. Multiple independent convection conditions (e.g.,
different surfaces with different $h$ values) can be registered simultaneously.

### Radiation Boundary Conditions

The `RadiationBC` class implements Stefan-Boltzmann radiation heat transfer on boundary
faces. The net radiative heat loss from a node is:

$$
Q_{\text{rad}} = -\sigma \, \varepsilon \, F \, A \, (T_i^4 - T_{\text{env}}^4)
$$

where $\sigma = 5.670 \times 10^{-8}$ W/m$^2$-K$^4$ is the Stefan-Boltzmann constant,
$\varepsilon$ is the surface emissivity, and $F$ is the geometric view factor. A
linearized radiation coefficient $h_{\text{rad}} = 4 \sigma \varepsilon T^3$ is provided
for stability analysis and time-step estimation.

### Fixed Temperature BC

The `FixedTemperatureBC` class enforces Dirichlet temperature constraints by overwriting
nodal temperatures to prescribed values after each solver step. It is applied last among
all boundary conditions so that the fixed constraint takes precedence. Multiple node
groups with different prescribed temperatures can be registered.

### Heat Flux BC

The `HeatFluxBC` class applies Neumann (prescribed flux) boundary conditions on boundary
faces. For each boundary node with area $A$, the heat input is $Q = q \, A$ where $q$
(W/m$^2$) is the prescribed flux, positive into the domain. Per-node or uniform boundary
areas are supported.

### Adiabatic Heating

The `AdiabaticHeating` class computes the Taylor-Quinney temperature rise from plastic
dissipation in high-rate deformation:

$$
\Delta T = \frac{\eta \, \sigma_{\text{eq}} \, \Delta\varepsilon^p}{\rho \, c_p}
$$

where $\eta$ is the Taylor-Quinney coefficient (default 0.9 for metals), $\sigma_{\text{eq}}$
is the von Mises stress, and $\Delta\varepsilon^p$ is the plastic strain increment. A
power-based variant accepting plastic power $\dot{W}^p = \sigma : \dot{\varepsilon}^p$
over a time step is also available.

### Thermal Time Step Control

The `ThermalTimeStep` class computes the stable time step for the explicit forward-Euler
thermal scheme:

$$
\Delta t_{\text{therm}} = \frac{h^2 \, \rho \, c_p}{2 \, k}
$$

where $h$ is the minimum element characteristic length. The thermal time step is
typically much larger than the mechanical CFL limit, so the class also computes the
recommended number of thermal subcycles per mechanical step via
$N_{\text{sub}} = \lceil \Delta t_{\text{mech}} / \Delta t_{\text{therm}} \rceil$.

### Coupled Thermo-Mechanical

The `CoupledThermoMechanical` class manages staggered (isothermal-split) coupling between
the mechanical and thermal solvers. Each coupled time step follows the sequence:

1. **Mechanical step** with the current temperature field.
2. **Adiabatic heating** from plastic work is buffered and added to nodal temperatures.
3. **Thermal sub-stepping** with $N_{\text{sub}}$ forward-Euler steps, applying
   conduction, convection, radiation, heat flux, and fixed-temperature BCs at each
   sub-step.
4. **Material property update** from the new temperature field (e.g., temperature-dependent
   yield stress or thermal conductivity).

Energy conservation is monitored via a per-step energy balance diagnostic.
