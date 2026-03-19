(part3)=
# Part III — Element Library

(ch08_elements)=
## Finite Elements

The NexusSim element library provides twenty-three element formulations spanning three-dimensional
solids, two-dimensional shells, one-dimensional beams and trusses, discrete spring and
damper elements, and extended formulations including thick shells, Kirchhoff elements,
plane/axisymmetric elements, and advanced locking-free solid and shell formulations. All
element implementations reside in the `discretization/` directory and inherit from the
abstract base class `nxs::physics::Element`. The element library spans three development waves:

- **Base** (10 elements): Hex8, Hex20, Tet4, Tet10, Wedge6, Shell4, Shell3, Beam2, Truss, Spring/Damper
- **Wave 15** (7 elements): ThickShell8, ThickShell6, DKT, DKQ, Plane, Axisymmetric, Connector
- **Wave 20** (6 elements): Belytschko-Tsay shell, Pyramid5, MITC4, EAS Hex8, B-bar Hex8, Isogeometric shell

### Element Base Class

The `Element` class (`physics/element.hpp`) defines the interface that all element types
must implement. This design permits the solver to treat all elements polymorphically while
retaining the option of static dispatch for performance-critical GPU kernels.

```cpp
class Element {
public:
    struct Properties {
        ElementType type;
        ElementTopology topology;
        int num_nodes;
        int num_gauss_points;
        int num_dof_per_node;
        int spatial_dim;
    };

    virtual Properties properties() const = 0;
    virtual void shape_functions(const Real xi[3], Real* N) const = 0;
    virtual void shape_derivatives(const Real xi[3], Real* dN) const = 0;
    virtual Real jacobian(const Real xi[3], const Real* coords, Real* J) const = 0;
    virtual void strain_displacement_matrix(const Real xi[3],
                                            const Real* coords, Real* B) const = 0;
    virtual void gauss_quadrature(Real* points, Real* weights) const = 0;
    virtual void mass_matrix(const Real* coords, Real density, Real* M) const = 0;
    virtual void stiffness_matrix(const Real* coords, Real E, Real nu,
                                  Real* K) const = 0;
    virtual void internal_force(const Real* coords, const Real* disp,
                                const Real* stress, Real* fint) const = 0;
    virtual Real volume(const Real* coords) const = 0;
    virtual Real characteristic_length(const Real* coords) const = 0;
    virtual bool contains_point(const Real* coords, const Real* point,
                                Real* xi) const = 0;
};
```

All shape function and internal force methods are annotated with
`KOKKOS_INLINE_FUNCTION` for GPU portability.

### Element Factory

The `ElementFactory` class provides static methods for creating element instances:

```cpp
std::unique_ptr<Element> ElementFactory::create(ElementType type);
ElementType ElementFactory::from_string(const std::string& type_str);
```

### Element Summary

The following table summarizes all available element types:

| Element       | Nodes | DOFs | Gauss Points | Shape Functions       | Topology     |
|---------------|-------|------|--------------|-----------------------|--------------|
| Hex8          | 8     | 24   | 8 (2³)       | Trilinear             | Hexahedron   |
| Hex20         | 20    | 60   | 27 (3³)      | Serendipity quadratic | Hexahedron   |
| Tet4          | 4     | 12   | 1            | Linear                | Tetrahedron  |
| Tet10         | 10    | 30   | 4            | Quadratic             | Tetrahedron  |
| Wedge6        | 6     | 18   | 6 (2×3)      | Linear prismatic      | Wedge        |
| Shell4        | 4     | 24   | 4 (2²)       | Bilinear              | Quadrilateral|
| Shell3        | 3     | 18   | 1 or 3       | Linear                | Triangle     |
| Beam2         | 2     | 12   | 2            | Linear / Hermite      | Line         |
| Truss         | 2     | 6    | 1            | Linear                | Line         |
| Spring/Damper | 2     | 6    | N/A          | N/A                   | Discrete     |
| ThickShell8   | 8     | 24   | 8 (2³)       | Trilinear             | Hexahedron   |
| ThickShell6   | 6     | 18   | 6 (2×3)      | Linear prismatic      | Wedge        |
| DKT Shell     | 3     | 9    | 3            | Discrete Kirchhoff    | Triangle     |
| DKQ Shell     | 4     | 12   | 4 (2²)       | Discrete Kirchhoff    | Quadrilateral|
| Plane         | 3–4   | 6–8  | 1–4          | Linear/Bilinear       | Tri3/Quad4   |
| Axisymmetric  | 3–4   | 6–8  | 1–4          | Linear/Bilinear       | Tri3/Quad4   |
| Connector     | 2     | 12   | N/A          | N/A                   | Discrete     |

### Global Shape Function Derivatives

For all isoparametric solid elements, the physical (global) derivatives of the shape
functions are obtained from the natural (parametric) derivatives through the Jacobian
transformation:

$$
\frac{\partial N_i}{\partial x_j} = \left(\mathbf{J}^{-T}\right)_{jk}
\frac{\partial N_i}{\partial \xi_k}
$$

where the Jacobian matrix is defined as:

$$
J_{ij} = \frac{\partial x_i}{\partial \xi_j}
= \sum_{k=1}^{n_{\text{nodes}}} x_k^{(i)} \frac{\partial N_k}{\partial \xi_j}
$$

:::{note}
The use of $\mathbf{J}^{-T}$ (the inverse transpose) is essential. An earlier version
of the code incorrectly used $\mathbf{J}^{-1}$, which produced correct results only for
axis-aligned hexahedral meshes (where $\mathbf{J}$ is diagonal and therefore symmetric).
For general meshes — particularly tetrahedral Kuhn decompositions of hexahedral domains —
the error ranged from 50% to 700%. This was corrected in all five solid element types
(Hex8, Hex20, Tet4, Tet10, Wedge6).
:::

### Element Stiffness Matrix

For all continuum elements, the element stiffness matrix is computed by numerical
integration:

$$
\mathbf{K}_e = \int_{\Omega_e} \mathbf{B}^T \mathbf{D} \, \mathbf{B} \,
\mathrm{d}\Omega
\approx \sum_{g=1}^{n_{\text{gp}}} \mathbf{B}_g^T \, \mathbf{D} \, \mathbf{B}_g \,
\det(\mathbf{J}_g) \, w_g
$$

where $\mathbf{B}$ is the strain–displacement matrix, $\mathbf{D}$ is the constitutive
matrix, $\mathbf{J}_g$ is the Jacobian at Gauss point $g$, and $w_g$ is the
corresponding quadrature weight.

---

### Hex8 — 8-Node Hexahedron

The Hex8 element is the standard trilinear hexahedral element with eight corner nodes.
It is the most commonly used solid element and is suitable for meshes with regular
geometry.

**Shape functions:**

$$
N_i(\xi, \eta, \zeta) = \frac{1}{8}(1 + \xi_i \xi)(1 + \eta_i \eta)(1 + \zeta_i \zeta)
$$

where $(\xi_i, \eta_i, \zeta_i) \in \{-1, +1\}^3$ are the natural coordinates of node $i$.

**Integration scheme:** Full integration with $2 \times 2 \times 2 = 8$ Gauss points.

**Key methods:**

| Method                           | Description                                |
|----------------------------------|--------------------------------------------|
| `shape_functions(xi, N)`         | Evaluates 8 shape function values          |
| `shape_derivatives(xi, dN)`      | Computes $8 \times 3$ parametric derivatives |
| `shape_derivatives_global(xi, coords, dN_global, detJ)` | Physical derivatives via $\mathbf{J}^{-T}$ |
| `stiffness_matrix(coords, E, nu, K)` | Assembles $24 \times 24$ element stiffness |
| `mass_matrix(coords, rho, M)`    | Assembles $24 \times 24$ consistent mass   |
| `internal_force(coords, disp, stress, fint)` | Computes 24-component force vector |

**Known limitations:** Susceptible to shear locking for bending-dominated problems in
thin structures. Use Hex20 or Shell4 elements in such cases.

---

### Hex20 — 20-Node Serendipity Hexahedron

The Hex20 element extends the Hex8 with mid-edge nodes, providing quadratic interpolation
within each element. It is well-suited for problems requiring higher accuracy, such as
stress concentrations and curved boundaries.

**Nodes:** 8 corner nodes + 12 mid-edge nodes = 20 nodes, 60 DOFs.

**Integration scheme:** Full integration with $3 \times 3 \times 3 = 27$ Gauss points.

**Shape functions:** Serendipity quadratic functions — corner nodes use the product
$(1 + \xi_i\xi)(1 + \eta_i\eta)(1 + \zeta_i\zeta)(\xi_i\xi + \eta_i\eta + \zeta_i\zeta - 2)/8$,
while mid-edge nodes use reduced forms.

---

### Tet4 — 4-Node Tetrahedron

The Tet4 is the simplest three-dimensional element. It uses linear shape functions
expressed in volume (barycentric) coordinates:

$$
N_1 = 1 - \xi - \eta - \zeta, \quad
N_2 = \xi, \quad
N_3 = \eta, \quad
N_4 = \zeta
$$

**Integration scheme:** Single Gauss point at the centroid.

**Properties:** The Tet4 is a constant-strain element — the strain is uniform throughout
the element. This makes it overly stiff for bending problems (volumetric locking) and
requires fine meshes for accurate results.

---

### Tet10 — 10-Node Tetrahedron

The Tet10 element adds six mid-edge nodes to the Tet4, yielding quadratic interpolation.
It produces significantly more accurate results than the Tet4, particularly for
bending-dominated problems.

**Nodes:** 4 corner + 6 mid-edge = 10 nodes, 30 DOFs.

**Integration scheme:** 4 Gauss points.

---

### Wedge6 — 6-Node Wedge

The Wedge6 (pentahedral) element has a triangular cross-section with linear interpolation.
It arises naturally in transitional regions between hexahedral and tetrahedral meshes.

**Nodes:** 6 nodes, 18 DOFs.

**Integration scheme:** $2 \times 3 = 6$ Gauss points (2 through thickness × 3 in
triangular cross-section).

**Shape functions:** Products of triangular area coordinates and linear functions in
the prismatic direction.

---

### Shell4 — 4-Node Quadrilateral Shell

The Shell4 element models thin-walled structures with a 6-DOF formulation: three
translational DOFs ($u$, $v$, $w$) and three rotational DOFs ($\theta_x$, $\theta_y$,
$\theta_z$) per node.

**Total DOFs:** $4 \times 6 = 24$ per element.

**Integration scheme:** $2 \times 2 = 4$ Gauss points in the mid-surface.

**Stiffness decomposition:**

The element stiffness matrix is assembled from three contributions:

$$
\mathbf{K}_{\text{local}} = \mathbf{K}_{\text{membrane}}
+ \mathbf{K}_{\text{bending}} + \mathbf{K}_{\text{shear}}
$$

where the membrane stiffness uses the in-plane strain–displacement matrix, the bending
stiffness uses the curvature–displacement matrix, and the shear stiffness is based on
Reissner–Mindlin theory.

**Local-to-global transformation:** The element stiffness in global coordinates is
obtained as:

$$
\mathbf{K}_{\text{global}} = \mathbf{T}^T \mathbf{K}_{\text{local}} \, \mathbf{T}
$$

where $\mathbf{T}$ is the $24 \times 24$ rotation matrix constructed from the element's
local coordinate system.

**Known limitation:** The Shell4 exhibits shear locking with full integration,
producing deflections that are approximately 1.5% of the Euler–Bernoulli beam theory
prediction for thin shells. This is a known limitation of the Reissner–Mindlin
formulation with full integration and is not a solver defect.

---

### Shell3 — 3-Node Triangular Shell

The Shell3 is a flat triangular shell element with 6 DOFs per node (18 total). It
supports 1-point (centroid) or 3-point Gauss integration. The Shell3 arises in meshes
generated from surface triangulations.

---

### Beam2 — 2-Node Beam

The Beam2 is a two-node beam element with 6 DOFs per node (12 total): three translations
and three rotations. It uses linear interpolation for axial behavior and Hermite cubic
interpolation for bending, following Euler–Bernoulli or Timoshenko beam theory.

**Integration scheme:** 2 Gauss points along the beam axis.

---

### Truss — 2-Node Truss

The Truss element connects two nodes and resists only axial force. Each node has 3
translational DOFs (6 total). A single integration point is used.

**Force computation:** $F = E A \, \varepsilon$, where $A$ is the cross-sectional area
and $\varepsilon = (\ell - \ell_0)/\ell_0$ is the engineering strain.

---

### Spring and Damper Elements

Discrete elements connecting two nodes with 3 translational DOFs per node (6 total).
No integration points are used; forces are computed directly from the relative
displacement or velocity of the two nodes.

**Spring:** $F = k \, \Delta x$ (linear stiffness)

**Damper:** $F = c \, \Delta v$ (viscous damping)

**Spring-Damper:** $F = k \, \Delta x + c \, \Delta v$ (combined)

All three variants support nonlinear force–displacement or force–velocity curves via the
load curve system (see {ref}`Section 18.1 <ch18_loads>`).

---

### Wave 15: Extended Element Formulations

The following additional elements are available in `physics/elements_wave15.hpp`
(namespace `nxs::elements`):

**ThickShell8** — 8-node thick shell element with a hex-like topology. Uses full 3D stress
evaluation ($\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$, $\sigma_{xy}$, $\sigma_{yz}$,
$\sigma_{xz}$) with $2 \times 2 \times 2$ Gauss integration. Suitable for structures
where through-thickness stress cannot be neglected.

**ThickShell6** — 6-node thick shell element with a wedge-shaped topology (triangular
in-plane, linear through thickness). Uses $2 \times 3$ integration. Fills the same role
as ThickShell8 for triangular mesh regions.

**DKT Shell** — Discrete Kirchhoff Triangle with 3 nodes and 9 DOFs (one transverse
displacement and two rotations per node). Enforces the Kirchhoff constraint (zero
transverse shear) discretely at selected points, avoiding shear locking entirely. Best
suited for thin plate and shell bending problems.

**DKQ Shell** — Discrete Kirchhoff Quadrilateral with 4 nodes and 12 DOFs. Extends the
DKT concept to quadrilateral geometry using $2 \times 2$ integration. Provides improved
accuracy over DKT for regular meshes.

**Plane Element** — 2D element for plane-stress and plane-strain analysis. Supports both
Quad4 (4 nodes, bilinear) and Tri3 (3 nodes, linear) topologies. The stress state flag
selects between plane stress ($\sigma_{zz} = 0$) and plane strain ($\varepsilon_{zz} = 0$).

**Axisymmetric Element** — Models solids of revolution using a 2D cross-section with an
additional hoop strain component $\varepsilon_{\theta\theta} = u_r / r$. Supports Quad4
and Tri3 topologies. Integration weights include the $2\pi r$ circumferential factor.

**Connector Element** — 2-node discrete element with 6 DOFs per node (12 total),
representing spot welds, rivets, or bolts. Transmits forces and moments between two
nodes with configurable stiffness in all six DOF directions. Supports failure criteria
based on normal force, shear force, and moment resultants.

---

### Wave 20: Advanced Element Formulations

The following additional elements are available in `discretization/elements_wave20.hpp`
(namespace `nxs::elements`):

**Belytschko-Tsay Shell** — High-performance 4-node shell element with one-point in-plane
integration and Belytschko-Tsay hourglass stabilization. The single-point integration
reduces cost by a factor of 4 compared to full integration, making it the element of
choice for large-scale explicit crash simulations. Hourglass control uses a combination
of viscous and stiffness-based stabilization to suppress zero-energy modes.

**Pyramid5** — 5-node pyramid element for transitioning between hexahedral and tetrahedral
mesh regions. Uses 5 shape functions with a collapsed hexahedral topology. The degenerate
apex requires special treatment of the Jacobian singularity.

**MITC4** — Mixed Interpolation of Tensorial Components shell element. Eliminates shear
locking by independently interpolating the transverse shear strain field using tying
points rather than direct interpolation from nodal rotations. Provides accurate results
for both thin and thick shell regimes.

**EAS Hex8** — Enhanced Assumed Strain 8-node hexahedron. Augments the standard displacement
field with internal (element-level) enhanced strain modes that are condensed out at the
element level. Eliminates volumetric locking and improves bending accuracy without the
cost of higher-order elements.

**B-bar Hex8** — Selective reduced integration hexahedron using the $\bar{B}$ (B-bar)
method. The volumetric part of the strain-displacement matrix is replaced with a
volume-averaged value, preventing volumetric locking for nearly incompressible materials
while retaining full-integration accuracy for the deviatoric response.

**Isogeometric Shell** — NURBS-based shell element that uses the same basis functions as
the CAD geometry representation. Provides exact geometry representation and higher
inter-element continuity ($C^{p-1}$ for degree $p$ NURBS) compared to conventional
Lagrangian elements. Supports arbitrary polynomial degree with knot-span-based integration.
