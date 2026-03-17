(part6)=
# Part VI — Contact and Constraints

This part describes the contact mechanics formulations, rigid body dynamics and kinematic
constraints, and the load and boundary condition subsystems.

---

(ch16_contact)=
## Contact Mechanics

The contact module (`fem/` directory) provides sixteen contact formulations. These are
organized from the simplest penalty-based methods to the most sophisticated mortar and
tied formulations.

### Contact Types

| Contact Type       | Method              | Governing Relation                            |
|--------------------|---------------------|-----------------------------------------------|
| Penalty            | Node-to-surface     | $F = k_n \cdot g$ (gap function)              |
| Node-to-Surface    | Projection          | Node projected onto master surface             |
| Surface-to-Surface | Segment-based       | Full segment pair interaction                  |
| Hertzian           | Analytical          | $F = k \, \delta^{3/2}$ (sphere–sphere)       |
| Mortar             | Segment-to-segment  | D/M integrals, augmented Lagrangian            |
| Tied               | Penalty/constraint  | Permanent bonding with optional failure        |
| Voxel Collision    | Grid-based          | Spatial hashing for broad-phase detection      |

### Contact Interface

All contact types implement the following base interface:

```cpp
class Contact {
public:
    void detect(const Mesh& mesh, const State& state);
    void enforce(State& state, Real dt);
    void set_penalty_stiffness(Real k);
    void set_friction(Real mu_s, Real mu_d);
};
```

### Friction Model

The `FrictionModel` class (`fem/friction_model.hpp`) implements Coulomb friction with
exponential decay from static to dynamic coefficients:

$$
F_t = \min\!\left(\mu \, F_n, \, \|F_{\text{trial}}\|\right)
$$

where:

$$
\mu = \mu_d + (\mu_s - \mu_d) \, e^{-\alpha \|\dot{u}_t\|}
$$

```cpp
class FrictionModel {
    Real mu_static;
    Real mu_dynamic;
    Real decay_coefficient;
};
```

### Hertzian Contact

The `HertzianContact` class (`fem/hertzian_contact.hpp`) implements analytical Hertz
contact theory for sphere–sphere or sphere–plane interactions:

$$
F = \frac{4}{3} E^* \sqrt{R^*} \, \delta^{3/2}
$$

where the effective modulus and radius are:

$$
\frac{1}{E^*} = \frac{1-\nu_1^2}{E_1} + \frac{1-\nu_2^2}{E_2}, \qquad
\frac{1}{R^*} = \frac{1}{R_1} + \frac{1}{R_2}
$$

Contact damping follows the Hunt–Crossley model:

$$
F_d = \alpha \, \delta^{3/2} \, \dot{\delta}
$$

where $\alpha$ is derived from the coefficient of restitution.

### Mortar Contact

The `MortarContact` class (`fem/mortar_contact.hpp`) implements a segment-to-segment
mortar formulation with augmented Lagrangian enforcement:

1. **Segment clipping** — Polygon clipping determines the overlapping region between
   slave and master segments.
2. **D/M integrals** — The mortar integrals $D_{ij}$ and $M_{ij}$ are evaluated over
   the clipped region using Gauss quadrature.
3. **Contact stiffness** — The contact contribution to the global stiffness matrix is
   assembled from the mortar integrals.
4. **Augmented Lagrangian update** — The Lagrange multipliers are updated iteratively:
   $\lambda_{n+1} = \lambda_n + \epsilon_n \, g_n$

### Tied Contact

The `TiedContact` class (`fem/tied_contact.hpp`) enforces permanent bonding between
slave and master surfaces using penalty stiffness:

```cpp
class TiedContact {
    void tie_surfaces(const std::vector<Index>& slave,
                      const std::vector<Index>& master);
    void set_failure_stress(Real sigma_n, Real sigma_s);
    void set_failure_strain(Real eps_n, Real eps_s);
    void check_failure(const State& state);
};
```

**Tied-with-failure:** When failure stresses or strains are specified, the tied
constraint monitors the interface forces. When the normal or shear stress exceeds the
specified threshold, the bond is broken and the nodes are released.

### Voxel Collision Detection

The `VoxelCollision` class (`fem/voxel_collision.hpp`) provides a grid-based
broad-phase collision detection algorithm using spatial hashing. It is designed for
efficient detection of potential contact pairs before the narrow-phase contact algorithms
are applied.

### Wave 12: Extended Contact Formulations

The following additional contact capabilities are available in `fem/contact_wave12.hpp`
(namespace `nxs::fem`):

**Contact Stiffness Scaler** — Adaptive penalty stiffness that scales automatically based
on the local element size, material modulus, and time step. Prevents both excessive
penetration (too soft) and numerical instability (too stiff).

**Velocity-Dependent Friction** — Extends the Coulomb model with an explicit static-to-kinetic
transition governed by relative sliding velocity: $\mu(v) = \mu_d + (\mu_s - \mu_d)\,e^{-\alpha v}$.
The decay coefficient $\alpha$ is user-configurable.

**Shell Thickness Contact** — Accounts for shell element thickness offsets when computing
contact gaps. The mid-surface position is offset by $t/2$ along the element normal so that
contact occurs at the physical outer surface rather than at the reference surface.

**Edge-to-Edge Contact** — Detects contact between beam or shell edge pairs using closest-point
projection between line segments. Computes gap, normal direction, and penalty force for
edge–edge interactions that node-to-surface methods miss.

**Segment-Based Contact** — Face-to-face penalty formulation using a bucket-sort spatial
search for broad-phase pair detection. Each segment pair is checked for overlap; penalty
forces are distributed to all contributing nodes.

**Self-Contact** — Single-surface contact detection where one surface can contact itself.
Uses spatial hashing to identify candidate segment pairs and excludes adjacent elements
to avoid false positives.

**Symmetric Contact** — Bidirectional contact where both surfaces are simultaneously treated
as master and slave. Averaging of the two pass results eliminates the master/slave bias
present in standard one-pass algorithms.

**Rigid-Deformable Contact** — Specialized interface between rigid bodies and deformable
elements. Rigid surface nodes are treated as prescribed-motion master nodes, and contact
forces are accumulated into the rigid body force/moment resultants.

**Multi-Surface Contact Manager** — Automatic surface detection and dispatch layer that
assigns contact pairs to the appropriate algorithm based on surface type (shell, solid,
rigid, beam). Manages a global contact table and invokes detection/enforcement for all
active pairs each time step.

---

(ch17_rigid)=
## Rigid Bodies and Constraints

### Rigid Bodies

The `RigidBody` class (`fem/rigid_body.hpp`) models rigid body dynamics using the
Newton–Euler equations:

$$
M \, \mathbf{a}_{\text{cm}} = \mathbf{F}, \qquad
\mathbf{I} \, \boldsymbol{\alpha} = \boldsymbol{\tau}
$$

where $M$ is the total mass, $\mathbf{a}_{\text{cm}}$ is the center-of-mass acceleration,
$\mathbf{I}$ is the inertia tensor, and $\boldsymbol{\alpha}$ is the angular acceleration.

```cpp
class RigidBody {
    void set_mass(Real m);
    void set_inertia(const Mat3& I);
    void set_center_of_mass(const Vec3r& cm);
    void compute_rigid_motion(Real dt);
    void update_node_positions();
    Vec3r angular_momentum() const;
    Vec3r angular_velocity() const;
};
```

### Rigid Walls

The `RigidWall` class (`fem/rigid_wall.hpp`) provides four wall geometries for
constraining node motion:

| Wall Type    | Geometry                     | Application              |
|--------------|------------------------------|--------------------------|
| Planar       | Infinite plane with normal   | Simple barriers          |
| Cylindrical  | Cylinder with axis and radius| Pipes, containment       |
| Spherical    | Sphere with center and radius| Hemispherical dies       |
| Moving       | Prescribed-velocity wall     | Crush simulations        |

```cpp
class RigidWall {
    void set_wall_type(WallType type);
    void set_velocity(const Vec3r& v);
    Real compute_gap(const Vec3r& point) const;
    Vec3r compute_normal(const Vec3r& point) const;
    void enforce(State& state, Real dt);
};
```

### Kinematic Constraints

The `Constraints` class (`fem/constraints.hpp`) implements multi-point constraints and
joints:

| Constraint       | Description                                             |
|------------------|---------------------------------------------------------|
| RBE2             | Rigid spider: slave nodes follow master rigidly         |
| RBE3             | Interpolation: master DOFs = weighted average of slaves |
| Revolute joint   | One rotational DOF free                                 |
| Spherical joint  | Three rotational DOFs free                              |
| Cylindrical joint| One rotation + one translation free                     |

```cpp
class Constraints {
    void add_rbe2(Index master, const std::vector<Index>& slaves);
    void add_rbe3(Index master, const std::vector<Index>& slaves,
                  const std::vector<Real>& weights);
    void add_joint(JointType type, Index node_a, Index node_b,
                   const Vec3r& axis);
    void enforce(State& state, Real dt);
};
```

---

(ch18_loads)=
## Loads and Boundary Conditions

### Load Curves

The `LoadCurve` class (`fem/load_curve.hpp`) provides piecewise-linear time functions
for scaling loads:

```cpp
class LoadCurve {
    void add_point(Real time, Real value);
    Real evaluate(Real time) const;
    void set_extrapolation(Extrapolation mode);
};
```

**Extrapolation modes:** `Zero` (return 0 outside range), `Constant` (hold last value),
`Linear` (continue slope).

The `LoadCurveManager` class provides a centralized registry:

```cpp
class LoadCurveManager {
    Index add_curve(const LoadCurve& curve);
    Real evaluate(Index curve_id, Real time);
};
```

### Load Types

| Load Type           | Description                                     |
|---------------------|-------------------------------------------------|
| Nodal force         | Point force at a node ($F_x, F_y, F_z$)        |
| Distributed load    | Force per unit length or area                   |
| Body force          | Acceleration field (e.g., gravity)              |
| Pressure            | Normal force per unit area on surfaces          |
| Prescribed motion   | Enforced displacement, velocity, or acceleration|

```cpp
class LoadManager {
    void add_nodal_force(Index node, int dof, Real value, Index curve_id = -1);
    void add_body_force(const Vec3r& acceleration);
    void add_pressure(const std::vector<Index>& face_nodes, Real pressure);
    void compute_external_forces(std::vector<Real>& F_ext, Real time);
};
```

### Initial Conditions

The `InitialConditions` class (`fem/initial_conditions.hpp`) specifies the state at
$t = 0$:

```cpp
class InitialConditions {
    void set_velocity(Index node, const Vec3r& v);
    void set_velocity_field(const std::vector<Index>& nodes, const Vec3r& v);
    void set_temperature(Index node, Real T);
    void apply(State& state);
};
```
