(part4)=
# Part IV — Constitutive Models

This part describes the material models, failure criteria, equations of state, and
composite analysis capabilities provided by NexusSim. The constitutive models are
implemented in the `physics/` directory and its subdirectories `physics/failure/` and
`physics/composite/`.

---

(ch09_materials)=
## Material Models

NexusSim provides fifty-four continuum material models for finite element analysis and
three additional material models for peridynamics. All FEM material models share a common
interface and data structure. The material library spans four development waves:

- **Wave 1** (14 models): Core elasticity, plasticity, hyperelasticity, foam, and rate-dependent models
- **Wave 10** (20 models): Anisotropic plasticity, advanced rate models, cohesive zone, and specialty materials
- **Wave 18** (20 models): Tier 2 materials including explosive burn, creep, Drucker-Prager, and programmed detonation

### Material Data Structures

#### MaterialState

The `MaterialState` structure (`physics/material.hpp`) stores the current mechanical state
at each integration point:

```cpp
struct MaterialState {
    Real stress[6];                    // Voigt: σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz
    Real strain[6];                    // Voigt notation
    Real F[9];                         // Deformation gradient (row-major 3×3)
    Real history[32];                  // Material-specific history variables
    Real effective_plastic_strain;
    Real damage;
    Real temperature;
    bool failed;
};
```

The Voigt notation convention used throughout NexusSim is:

$$
\boldsymbol{\sigma} = \begin{bmatrix}
\sigma_{xx} & \sigma_{yy} & \sigma_{zz} & \sigma_{xy} & \sigma_{yz} & \sigma_{xz}
\end{bmatrix}^T
$$

$$
\boldsymbol{\varepsilon} = \begin{bmatrix}
\varepsilon_{xx} & \varepsilon_{yy} & \varepsilon_{zz} &
\gamma_{xy} & \gamma_{yz} & \gamma_{xz}
\end{bmatrix}^T
$$

where $\gamma_{ij} = 2\varepsilon_{ij}$ denotes the engineering shear strains.

#### MaterialProperties

The `MaterialProperties` structure contains all parameters for all material types. The
`type` field selects which parameters are active:

```cpp
struct MaterialProperties {
    physics::MaterialType type;
    Real E, nu, density;              // Basic elastic properties
    Real yield_stress;                // Von Mises / Johnson-Cook
    Real hardening_modulus;           // Isotropic hardening modulus
    // ... extended fields for all material types ...
};
```

### Constitutive Update Interface

Each material model implements the following function:

```cpp
void compute_stress(MaterialState& state, const MaterialProperties& props, Real dt);
```

**Parameters:**
- `state` — Current material state; updated in place with the new stress.
- `props` — Material properties (read-only).
- `dt` — Time step (used by rate-dependent models).

---

### Material Model Reference

#### Elastic (Linear Isotropic)

**Parameters:** $E$ (Young's modulus), $\nu$ (Poisson's ratio)

**Constitutive matrix (isotropic):**

$$
\mathbf{D} = \begin{bmatrix}
\lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
\lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}
$$

where the Lamé parameters are:

$$
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad
\mu = \frac{E}{2(1+\nu)}
$$

**Application:** General structural analysis, benchmark problems.

---

#### Von Mises (Elasto-Plastic with Isotropic Hardening)

**Parameters:** $\sigma_y$ (initial yield stress), $H$ (hardening modulus)

**Yield function:**

$$
f = \sigma_{\text{vm}} - \sigma_y(\bar{\varepsilon}^p)
$$

where the von Mises equivalent stress is:

$$
\sigma_{\text{vm}} = \sqrt{\frac{3}{2} \, s_{ij} s_{ij}}
$$

and $s_{ij} = \sigma_{ij} - \frac{1}{3}\sigma_{kk}\delta_{ij}$ is the deviatoric stress.

**Radial return algorithm:**

1. Compute trial stress: $\boldsymbol{\sigma}_{\text{trial}} = \boldsymbol{\sigma}_n + \mathbf{D} : \Delta\boldsymbol{\varepsilon}$
2. Evaluate yield: $f = \sigma_{\text{vm}}(\boldsymbol{\sigma}_{\text{trial}}) - \sigma_y(\bar{\varepsilon}^p)$
3. If $f > 0$ (plastic):
   - $\Delta\gamma = f \, / \, (3G + H)$
   - $\boldsymbol{\sigma} = \boldsymbol{\sigma}_{\text{trial}} - 3G \, \Delta\gamma \, \mathbf{n}$
   - $\bar{\varepsilon}^p \leftarrow \bar{\varepsilon}^p + \Delta\gamma$

where $\mathbf{n}$ is the unit normal to the yield surface.

**Application:** General metal plasticity.

---

#### Johnson-Cook (Viscoplastic)

**Parameters:** $A$, $B$, $n$, $C$, $m$, $\dot{\varepsilon}_0$, $T_{\text{melt}}$, $T_{\text{room}}$

**Flow stress:**

$$
\sigma_y = \left(A + B \, \bar{\varepsilon}^{p\,n}\right)
\left(1 + C \ln\frac{\dot{\varepsilon}}{\dot{\varepsilon}_0}\right)
\left(1 - T^{*m}\right)
$$

where the homologous temperature is:

$$
T^* = \frac{T - T_{\text{room}}}{T_{\text{melt}} - T_{\text{room}}}
$$

**Application:** High strain-rate metal deformation, impact, and ballistic analyses.

---

#### Neo-Hookean (Hyperelastic)

**Parameters:** $C_{10} = \mu/2$ (where $\mu$ is the shear modulus)

**Cauchy stress:**

$$
\boldsymbol{\sigma} = \frac{\mu}{J}(\mathbf{B} - \mathbf{I})
+ \frac{\lambda}{J} \ln(J) \, \mathbf{I}
$$

where $\mathbf{B} = \mathbf{F}\mathbf{F}^T$ is the left Cauchy–Green tensor and
$J = \det(\mathbf{F})$.

**Application:** Rubber, soft biological tissue, moderate-strain elastomers.

---

#### Mooney-Rivlin (Hyperelastic)

**Parameters:** $C_{10}$, $C_{01}$, $D_1$ (bulk compressibility)

**Strain energy density:**

$$
W = C_{10}(\bar{I}_1 - 3) + C_{01}(\bar{I}_2 - 3) + \frac{1}{D_1}(J - 1)^2
$$

where $\bar{I}_1$ and $\bar{I}_2$ are the isochoric invariants of $\mathbf{B}$.

**Application:** Rubber, elastomers, seals.

---

#### Ogden (Hyperelastic)

**Parameters:** $\mu_i$, $\alpha_i$ for $i = 1, \ldots, N$ (up to 6 terms)

**Strain energy density:**

$$
W = \sum_{i=1}^{N} \frac{2\mu_i}{\alpha_i^2}
\left(\bar{\lambda}_1^{\alpha_i} + \bar{\lambda}_2^{\alpha_i}
+ \bar{\lambda}_3^{\alpha_i} - 3\right)
$$

where $\bar{\lambda}_j$ are the isochoric principal stretches.

**Application:** Large-stretch rubber behavior with complex stress–strain curves.

---

#### Orthotropic (Anisotropic Linear Elastic)

**Parameters:** $E_1, E_2, E_3$; $G_{12}, G_{13}, G_{23}$; $\nu_{12}, \nu_{13}, \nu_{23}$

**Constitutive relation:** $\boldsymbol{\sigma} = \mathbf{C} \, \boldsymbol{\varepsilon}$
where $\mathbf{C}$ is the $6 \times 6$ orthotropic stiffness matrix.

**Application:** Fiber-reinforced composites, wood, rolled metals.

---

#### Piecewise-Linear Plasticity

**Parameters:** Yield table as pairs $(\bar{\varepsilon}^p, \sigma_y)$

The yield stress is interpolated from a user-supplied table of effective plastic strain
versus flow stress. Linear interpolation is used between data points.

**Application:** General metals with experimentally measured stress–strain curves.

---

#### Tabulated Material

**Parameters:** Stress–strain table with optional strain-rate dependence

A fully general tabulated material that supports stress–strain curves at multiple strain
rates. At each time step, the appropriate curve is selected based on the current strain
rate, and interpolation provides the flow stress.

**Application:** Materials characterized by experimental curves.

---

#### Foam (Low-Density)

**Parameters:** $E$, $\nu$, $\sigma_y$ (crush plateau), $\varepsilon_d$ (densification strain)

**Behavior:** The foam model implements a three-phase response: elastic, plateau
(constant stress), and densification (exponentially increasing stress). Densification
occurs when the volumetric strain approaches $\varepsilon_d$:

$$
\sigma_d = \sigma_y \cdot \frac{1}{1 - |\varepsilon_v| / \varepsilon_d}
$$

The denominator is guarded against values $\leq 0$ to prevent numerical overflow.

**Application:** Polymer foams, packaging, energy absorption.

---

#### Crushable Foam

**Parameters:** Compressive yield stress as a function of volumetric strain $\sigma_p(\varepsilon_v)$, tensile cutoff

The crushable foam model stores the absolute volumetric strain as a positive value in
`history[0]`. The compressive yield stress is interpolated from a user-supplied
crush curve.

**Application:** Metallic foams, structural energy absorbers.

---

#### Honeycomb (Orthotropic Crush)

**Parameters:** Normal stress $\sigma_N$, transverse stress $\sigma_T$, shear stress
$\sigma_S$, plus nonlinear crush curves for each direction

**Application:** Honeycomb sandwich cores, crash-box structures.

---

#### Viscoelastic (Prony Series)

**Parameters:** $G_\infty$ (long-term modulus), $G_i$ and $\beta_i$ for $i = 1, \ldots, N$ (Prony series terms)

The viscoelastic model uses a strain-increment formulation with the previous strain
state stored in `history[23–28]`.

**Relaxation function:**

$$
G(t) = G_\infty + \sum_{i=1}^{N} G_i \, e^{-\beta_i t}
$$

**Application:** Polymers, rubber, biological tissue.

---

#### Cowper-Symonds (Rate-Dependent Plasticity)

**Parameters:** $\sigma_y$ (static yield), $D$ and $q$ (rate parameters)

**Dynamic yield stress:**

$$
\sigma_y^{\text{dyn}} = \sigma_y^{\text{static}}
\left[1 + \left(\frac{\dot{\varepsilon}}{D}\right)^{1/q}\right]
$$

**Application:** Steel and aluminum at moderate to high strain rates.

---

#### Zhao (Rate-Dependent)

**Parameters:** $A$, $B$, $n$, $C_1$, $C_2$, $\dot{\varepsilon}_0$

**Application:** Impact problems with complex rate sensitivity.

---

#### Elastic-Plastic with Failure

**Parameters:** $\sigma_y$ (yield stress), $\varepsilon_{\text{fail}}$ (failure strain)

Combines Von Mises plasticity with a simple maximum plastic strain failure criterion.
When $\bar{\varepsilon}^p > \varepsilon_{\text{fail}}$, the element is flagged as failed
and its stress is set to zero.

---

#### Rigid

**Parameters:** $\rho$ (density only)

The rigid material has infinite stiffness; the element undergoes no deformation. It is
used for impactors, fixtures, and rigid body components.

---

#### Null

**Parameters:** $\rho$ (density only)

The null material carries no stress but contributes mass. It is used for mass-only
elements and Lagrangian markers.

---

#### Hill Anisotropic Plasticity

**Parameters:** $\sigma_y$ (yield stress), $R_{00}$, $R_{45}$, $R_{90}$ (Lankford R-values)

**Yield function:**

$$
F(\sigma_{22} - \sigma_{33})^2 + G(\sigma_{33} - \sigma_{11})^2
+ H(\sigma_{11} - \sigma_{22})^2 + 2L\sigma_{23}^2 + 2M\sigma_{13}^2
+ 2N\sigma_{12}^2 = \sigma_y^2
$$

where $F$, $G$, $H$, $L$, $M$, $N$ are derived from the R-values.

**Application:** Sheet metal forming with planar anisotropy.

---

#### Barlat Yld2000

**Parameters:** $\sigma_y$, $m$ (exponent), $\alpha_1$ through $\alpha_8$ (anisotropy coefficients)

**Yield function:**

$$
\Phi = |S_1' - S_2'|^m + |2S_2'' + S_1''|^m + |2S_1'' + S_2''|^m = 2\sigma_y^m
$$

where $S_i'$ and $S_i''$ are principal values of two linear transformations of the stress tensor, parameterized by the eight $\alpha$ coefficients.

**Application:** Advanced aluminum sheet forming with strong in-plane anisotropy.

---

#### Tabulated Johnson-Cook

**Parameters:** Strain-rate table $\sigma_y(\bar{\varepsilon}^p, \dot{\varepsilon})$, temperature table $\sigma_y(\bar{\varepsilon}^p, T)$

The flow stress is determined by bilinear interpolation from user-supplied tables of
effective plastic strain versus flow stress at multiple strain rates and temperatures,
rather than using the analytical Johnson-Cook equation.

**Application:** Metals with experimentally measured rate and temperature sensitivity.

---

#### Concrete (Drucker-Prager Cap)

**Parameters:** $\alpha$ (friction angle), $\lambda$ (cap eccentricity), $f_c'$ (compressive strength), $f_t$ (tensile strength)

**Yield surface:**

$$
f_s = \sqrt{J_2} - \alpha \, I_1 - k = 0, \qquad
f_c = \sqrt{J_2^2 + (I_1 - L)^2 / R^2} - F_c = 0
$$

The shear envelope ($f_s$) is intersected with an elliptical cap ($f_c$) to capture
pressure-dependent hardening and compaction.

**Application:** Concrete, rock, and cementitious materials under confinement.

---

#### Fabric

**Parameters:** $E_1$, $E_2$ (warp/weft moduli), $G_{12}$, tensile-only flag

The fabric model carries biaxial tension with zero compressive stiffness. Stress is
set to zero when the corresponding normal strain is compressive.

**Application:** Airbags, parachutes, textile membranes.

---

#### Cohesive Zone

**Parameters:** $T_n^{\max}$ (normal traction), $T_s^{\max}$ (shear traction), $\delta_n^c$ (normal separation), $\delta_s^c$ (shear separation)

**Traction-separation law:**

$$
T_n = T_n^{\max} \frac{\delta_n}{\delta_n^c}
\exp\!\left(1 - \frac{\delta_n}{\delta_n^c}\right)
$$

**Application:** Delamination, adhesive joints, interfacial fracture.

---

#### Soil Cap

**Parameters:** $\alpha$ (Drucker-Prager slope), $W$ (cap hardening parameter), $D$ (cap hardening exponent), $R$ (cap ratio)

A cap plasticity model for geomaterials combining a linear Drucker-Prager shear
surface with a hardening elliptical cap. The cap expands with plastic volumetric
compaction.

**Application:** Soils, granular fill, buried structure analyses.

---

#### User-Defined Material

**Parameters:** User-supplied callback function pointer, up to 48 material constants

The user-defined material provides a callback interface for custom constitutive models.
The user implements `void user_material(MaterialState&, const Real* params, Real dt)`.

**Application:** Research materials, proprietary constitutive models.

---

#### Arruda-Boyce (8-Chain Rubber)

**Parameters:** $\mu$ (initial shear modulus), $\lambda_m$ (locking stretch), $D$ (bulk compressibility)

**Strain energy density:**

$$
W = \mu \sum_{i=1}^{5} \frac{C_i}{\lambda_m^{2i-2}}
(\bar{I}_1^i - 3^i)
+ \frac{1}{D}(J - 1)^2
$$

where $C_i$ are coefficients derived from the inverse Langevin function expansion.

**Application:** Rubber and elastomers with limited chain extensibility.

---

#### Shape Memory Alloy

**Parameters:** $E_A$, $E_M$ (austenite/martensite moduli), $\sigma_s^{AM}$, $\sigma_f^{AM}$ (forward transformation stresses), $\sigma_s^{MA}$, $\sigma_f^{MA}$ (reverse), $\varepsilon_L$ (max transformation strain)

The model tracks the martensite volume fraction $\xi$ and implements the superelastic
loading-unloading hysteresis with full strain recovery upon unloading.

**Application:** Superelastic NiTi devices, biomedical stents, seismic dampers.

---

#### Rate-Dependent Foam

**Parameters:** Loading curves $\sigma(\varepsilon, \dot{\varepsilon})$, unloading hysteresis factor

Stress-strain response is interpolated from tabulated loading curves at multiple strain
rates. Unloading follows a separate hysteretic path scaled by the user-specified factor.

**Application:** Automotive padding, helmet liners, energy absorbers at varying impact speeds.

---

#### Prony Viscoelastic

**Parameters:** $G_\infty$, $K_\infty$, pairs $(g_i, \tau_i^G)$ and $(k_i, \tau_i^K)$ for $i = 1, \ldots, N$

**Relaxation moduli:**

$$
G(t) = G_\infty + G_0 \sum_{i=1}^{N} g_i \, e^{-t/\tau_i^G}, \qquad
K(t) = K_\infty + K_0 \sum_{i=1}^{N} k_i \, e^{-t/\tau_i^K}
$$

Both deviatoric and volumetric relaxation are modeled via independent Prony series
branches with recursive convolution integration.

**Application:** Polymer melts, adhesives, encapsulants with multiple relaxation timescales.

---

#### Thermal Elastic-Plastic

**Parameters:** $E(T)$, $\sigma_y(T)$, $H(T)$, $\alpha_{\text{CTE}}(T)$ — temperature-dependent tables

Material properties are interpolated from temperature-dependent tables at the current
element temperature. Thermal strain $\varepsilon^T = \alpha \Delta T$ is subtracted
from the total strain before the mechanical constitutive update.

**Application:** Welding simulation, thermomechanical fatigue, hot forming.

---

#### Zerilli-Armstrong

**Parameters:** $C_0$, $C_1$, $C_3$, $C_4$, $C_5$, $n$ (BCC form); $C_0$, $C_2$, $C_3$, $C_4$ (FCC form)

**Flow stress (BCC):**

$$
\sigma_y = C_0 + C_1 \exp(-C_3 T + C_4 T \ln\dot{\varepsilon})
+ C_5 \bar{\varepsilon}^{p\,n}
$$

**Application:** Rate-dependent metals (steel BCC, copper/aluminum FCC) under high strain rate.

---

#### Steinberg-Guinan

**Parameters:** $Y_0$, $Y_{\max}$, $\beta$, $n$, $G_0$, $G_P'$, $G_T'$, $T_m$

**Flow stress:**

$$
\sigma_y = Y_0 \left[1 + \beta(\bar{\varepsilon}^p + \varepsilon_i)\right]^n
\frac{G(P,T)}{G_0}, \qquad \sigma_y \leq Y_{\max}
$$

where $G(P,T)$ accounts for pressure and temperature dependence of the shear modulus.

**Application:** Metals at very high pressures (>10 GPa) and strain rates (>10$^5$/s).

---

#### MTS (Mechanical Threshold Stress)

**Parameters:** $\hat{\sigma}_a$, $\hat{\sigma}_i$, $\hat{\sigma}_\varepsilon$, $g_{0i}$, $g_{0\varepsilon}$, $p_i$, $q_i$, $A$, $b$, $k_b$

**Flow stress:**

$$
\sigma_y / G = \sigma_a / G
+ s_i(\dot{\varepsilon}, T) \, \hat{\sigma}_i / G_0
+ s_\varepsilon(\dot{\varepsilon}, T) \, \hat{\sigma}_\varepsilon / G_0
$$

where $s_i$ and $s_\varepsilon$ are Arrhenius-type scaling functions. The model is
grounded in dislocation thermodynamics.

**Application:** Metals with physically-based rate and temperature dependence (copper, tantalum).

---

#### Blatz-Ko with Mullins

**Parameters:** $\mu$ (shear modulus), $r$ (damage parameter), $\eta_{\max}$ (max softening)

**Strain energy density:**

$$
W = \frac{\mu}{2}\left(\frac{I_2}{I_3} + 2\sqrt{I_3} - 5\right)
$$

The Mullins effect is modeled via a damage variable that tracks the historical maximum
strain energy; subsequent loading below this threshold exhibits softened response.

**Application:** Filled rubber with stress-softening (Mullins) behavior.

---

#### Laminated Glass

**Parameters:** Glass $E_g$, $\nu_g$, $\sigma_{\text{fail}}^g$; PVB interlayer $E_{\text{PVB}}$, $\nu_{\text{PVB}}$

A composite material model with brittle glass outer plies and a hyperelastic PVB
interlayer. Glass failure is governed by a maximum principal stress criterion; post-failure
load is carried by the PVB interlayer and any intact glass plies.

**Application:** Automotive windshields, architectural glazing, blast-resistant windows.

---

#### Spot Weld

**Parameters:** $F_N$ (normal strength), $F_S$ (shear strength), $d$ (nugget diameter), failure flag

A beam-type element material that transmits forces between connected shell parts.
Failure is governed by a combined normal-shear interaction criterion:

$$
\left(\frac{f_N}{F_N}\right)^2 + \left(\frac{f_S}{F_S}\right)^2 \geq 1
$$

**Application:** Spot-welded automotive assemblies, joint failure prediction.

---

#### Rate-Dependent Composite

**Parameters:** $E_1(\dot{\varepsilon})$, $E_2(\dot{\varepsilon})$, $G_{12}(\dot{\varepsilon})$, $X_T(\dot{\varepsilon})$, $Y_T(\dot{\varepsilon})$, $S(\dot{\varepsilon})$

Elastic moduli and strength values are interpolated from strain-rate-dependent tables.
The constitutive update uses the orthotropic framework with rate-adjusted properties.

**Application:** Composites under impact and crash loading with rate-sensitive matrix behavior.

---

### Wave 18: Tier 2 Material Models

The following twenty material models are available in `physics/material_wave18.hpp`
(namespace `nxs::physics`):

#### Explosive Burn

Programmed detonation material with Chapman-Jouguet detonation pressure and a
burn fraction $F$ that ramps from 0 to 1 based on detonation arrival time. Couples
with JWL EOS for detonation product expansion.

**Application:** High-explosive charges, shaped charges, mining blast.

---

#### Porous Elastic

Linear elastic model with pressure-dependent bulk modulus for porous media. The
porosity $n$ evolves with volumetric strain, and effective stress is computed
via Terzaghi's principle.

**Application:** Saturated soils, biological tissue, porous ceramics.

---

#### Brittle Fracture

Rankine-type material with smeared cracking. Cracks initiate when the maximum
principal stress exceeds the tensile strength $f_t$; post-crack behavior follows
a linear softening curve governed by the fracture energy $G_f$.

**Application:** Glass, ceramics, mortar joints.

---

#### Creep

Norton power-law creep model:

$$
\dot{\varepsilon}^c = A \, \sigma_{\text{eq}}^n \, \exp(-Q / RT)
$$

where $A$ is the creep coefficient, $n$ is the stress exponent, $Q$ is the
activation energy, and $T$ is the absolute temperature.

**Application:** High-temperature turbine blades, nuclear components, solder joints.

---

#### Kinematic Hardening

Prager kinematic hardening model with a back-stress tensor $\boldsymbol{\alpha}$
that translates the yield surface in stress space:

$$
f = \|\mathbf{s} - \boldsymbol{\alpha}\| - \sqrt{2/3} \, \sigma_y = 0
$$

**Application:** Cyclic loading, low-cycle fatigue, Bauschinger effect.

---

#### Drucker-Prager (FEM)

Smooth Drucker-Prager yield surface with associated or non-associated flow:

$$
f = \sqrt{J_2} + \alpha \, I_1 - k = 0
$$

where $\alpha$ and $k$ are derived from the friction angle $\phi$ and cohesion $c$.

**Application:** Geomaterials, soils, rock masses, concrete.

---

#### Tabulated Composite

Orthotropic composite with stress-strain response defined by tabulated curves in
each material direction. Supports separate tension and compression curves.

**Application:** Composites with experimentally measured direction-dependent behavior.

---

#### Ply Degradation

Progressive ply-by-ply stiffness degradation model for laminated composites.
Failed plies have their moduli reduced according to user-specified degradation
factors, with load redistribution to intact plies.

**Application:** Impact damage in composite laminates, post-failure load paths.

---

#### Orthotropic Plastic

Anisotropic plasticity with orthotropic elastic behavior and Hill-type yield
surface. Separate yield stresses and hardening curves for each material direction.

**Application:** Rolled metals and drawn tubes with directional properties.

---

#### Pinching Material

Cyclic material model with stiffness and strength degradation, pinching
(narrowing of hysteresis loops), and unloading/reloading rules for seismic
analysis.

**Application:** Reinforced concrete frames, masonry, timber connections under seismic loading.

---

#### Frequency-Dependent Viscoelastic

Complex modulus representation $G^*(\omega) = G'(\omega) + i \, G''(\omega)$
with storage and loss moduli provided as tabulated functions of frequency.
Time-domain response is recovered via Prony series fitting.

**Application:** Vibration isolation, NVH analysis, polymer bushings.

---

#### Generalized Viscoelastic

Multi-branch Maxwell model with arbitrary number of relaxation branches for
both deviatoric and volumetric response. Each branch has independent relaxation
time and modulus fraction.

**Application:** Advanced polymer characterization with many relaxation timescales.

---

#### Phase Transformation

Austenite-martensite phase transformation model with kinetics governed by
temperature and stress state. Transformation strain and latent heat are
incorporated for coupled thermo-mechanical analysis.

**Application:** TRIP steels, quenching simulation, heat treatment.

---

#### Polynomial Hardening

Hardening law expressed as a polynomial in effective plastic strain:

$$
\sigma_y = \sum_{i=0}^{N} c_i \, (\bar{\varepsilon}^p)^i
$$

**Application:** Metals with complex hardening curves that are poorly fit by power laws.

---

#### Viscoplastic Thermal

Rate-dependent plasticity with full temperature coupling. Flow stress depends
on strain, strain rate, and temperature via independent multiplicative factors
with tabulated input.

**Application:** Hot forming, welding, thermomechanical processing.

---

#### Porous Brittle

Brittle material with porosity-dependent stiffness and strength. Effective
properties are reduced by the Mori-Tanaka scheme based on void volume fraction.

**Application:** Porous ceramics, trabecular bone, lightweight cementitious materials.

---

#### Anisotropic Crush Foam

Orthotropic foam with independent crush curves in three material directions.
Each direction has a separate plateau stress and densification strain, allowing
directional energy absorption design.

**Application:** Directional energy absorbers, honeycomb-like foams.

---

#### Spring Hysteresis

Nonlinear spring element material with user-defined loading and unloading
curves, producing rate-independent hysteretic behavior. The unloading path
is parameterized by the maximum historical displacement.

**Application:** Rubber mounts, seismic isolators, nonlinear connections.

---

#### Programmed Detonation

Detonation material with user-specified detonation point and velocity. The
burn fraction at each element is computed from the detonation wave arrival
time $t_a = d / D$ where $d$ is the distance from the detonation point and
$D$ is the detonation velocity.

**Application:** Multi-point initiation, shaped charges, blast sequence design.

---

#### Bonded Interface

Cohesive-zone-like material for bonded interfaces with mixed-mode
traction-separation law. The effective opening $\delta = \sqrt{\delta_n^2 + \delta_s^2}$
governs damage evolution with separate normal and shear critical openings.

**Application:** Adhesive joints, composite bonded repairs, delamination.

---

### Material Library

Predefined material presets are available in `physics/material_library.hpp`:

| Preset                | $E$ (GPa) | $\nu$ | $\rho$ (kg/m³) | $\sigma_y$ (MPa) |
|-----------------------|-----------|--------|-----------------|-------------------|
| Mild Steel            | 210       | 0.30   | 7,850           | 250               |
| Aluminum 6061-T6      | 69        | 0.33   | 2,700           | 276               |
| Titanium Ti-6Al-4V    | 114       | 0.34   | 4,430           | 880               |
| Rubber (Neo-Hookean)  | ~1.5      | 0.4999 | 1,100           | N/A               |
| Concrete              | 30        | 0.20   | 2,400           | N/A               |
| EPS Foam              | 0.005     | 0.10   | 30              | 0.1               |

---

(ch10_failure)=
## Failure Models

Twenty-eight failure models are provided in the `physics/failure/` directory. Each model
implements the `FailureModel` interface and operates on the `FailureState` structure.
The failure library spans three development waves:

- **Wave 2** (6 models): Hashin, Tsai-Wu, Chang-Chang, GTN, GISSMO, Tabulated
- **Wave 11** (12 models): J-C failure, Cockcroft-Latham, Lemaitre, Puck, FLD, Wilkins, Tuler-Butcher, max stress/strain, energy, Wierzbicki, fabric
- **Wave 19** (10 models): LaDeveze delamination, Hoffman, Tsai-Hill, RTCl, Mullins, spalling, HC_DSSE, adhesive joint, windshield, generalized energy

### Failure Data Structures

```cpp
struct FailureState {
    Real damage;           // 0 = intact, 1 = fully failed
    bool failed;
    int failure_mode;      // Model-specific failure mode index
    Real history[16];      // Model-specific history variables
};

class FailureModel {
public:
    virtual bool check_failure(const MaterialState& state,
                               FailureState& fstate) = 0;
    virtual void update_damage(const MaterialState& state,
                               FailureState& fstate, Real dt) = 0;
};
```

### Failure Model Reference

#### Hashin Failure Criteria

**Application:** Unidirectional fiber-reinforced composites.

The Hashin model evaluates four independent failure modes:

**Mode 1 — Fiber tension** ($\sigma_{11} > 0$):

$$
\left(\frac{\sigma_{11}}{X_T}\right)^2
+ \frac{\sigma_{12}^2 + \sigma_{13}^2}{S_{12}^2} \geq 1
$$

**Mode 2 — Fiber compression** ($\sigma_{11} < 0$):

$$
\left(\frac{\sigma_{11}}{X_C}\right)^2 \geq 1
$$

**Mode 3 — Matrix tension** ($\sigma_{22} + \sigma_{33} > 0$):

$$
\frac{(\sigma_{22} + \sigma_{33})^2}{Y_T^2}
+ \frac{\sigma_{23}^2 - \sigma_{22}\sigma_{33}}{S_{23}^2}
+ \frac{\sigma_{12}^2 + \sigma_{13}^2}{S_{12}^2} \geq 1
$$

**Mode 4 — Matrix compression** ($\sigma_{22} + \sigma_{33} < 0$):

$$
\left[\left(\frac{Y_C}{2S_{23}}\right)^2 - 1\right]
\frac{\sigma_{22} + \sigma_{33}}{Y_C}
+ \frac{(\sigma_{22} + \sigma_{33})^2}{4S_{23}^2}
+ \frac{\sigma_{12}^2 + \sigma_{13}^2}{S_{12}^2} \geq 1
$$

---

#### Tsai-Wu Failure Criterion

**Application:** General composite laminates.

The Tsai-Wu criterion is a single quadratic interaction formula:

$$
F_1\sigma_1 + F_2\sigma_2 + F_{11}\sigma_1^2 + F_{22}\sigma_2^2
+ F_{66}\sigma_{12}^2 + 2F_{12}\sigma_1\sigma_2 \geq 1
$$

where the strength parameters are:

$$
F_1 = \frac{1}{X_T} - \frac{1}{X_C}, \quad
F_2 = \frac{1}{Y_T} - \frac{1}{Y_C}, \quad
F_{11} = \frac{1}{X_T X_C}, \quad
F_{22} = \frac{1}{Y_T Y_C}, \quad
F_{66} = \frac{1}{S^2}
$$

The interaction term $F_{12}$ is typically taken as
$F_{12} = -\frac{1}{2}\sqrt{F_{11}F_{22}}$.

---

#### Chang-Chang Failure Criteria

**Application:** Laminated composites with in-situ strength effects.

The Chang-Chang model is a modified Hashin formulation that accounts for in-situ ply
strengths, which differ from unidirectional test coupon strengths due to constraint
effects.

---

#### GTN (Gurson-Tvergaard-Needleman)

**Application:** Ductile metal fracture.

**Yield function:**

$$
\Phi = \left(\frac{q}{\sigma_y}\right)^2
+ 2 q_1 f^* \cosh\!\left(\frac{3 q_2 \, p}{2 \sigma_y}\right)
- \left(1 + (q_1 f^*)^2\right) = 0
$$

where $q$ is the von Mises stress, $p$ is the hydrostatic pressure, $\sigma_y$ is the
matrix yield stress, and $f^*$ is the effective void volume fraction (modified from the
actual porosity $f$ to model void coalescence).

**Void evolution:** $\dot{f} = \dot{f}_{\text{growth}} + \dot{f}_{\text{nucleation}}$

---

#### GISSMO (Generalized Incremental Stress-State Dependent)

**Application:** General-purpose ductile damage model.

**Damage evolution:**

$$
\frac{\mathrm{d}D}{\mathrm{d}t}
= \frac{n}{\varepsilon_f(\eta, \dot{\varepsilon})}
\, D^{(n-1)/n} \, \dot{\varepsilon}^p
$$

where $n$ is the damage exponent and $\varepsilon_f$ is the failure strain, which depends
on the stress triaxiality $\eta$ and strain rate $\dot{\varepsilon}$.

---

#### Tabulated Failure

**Application:** Materials characterized by experimental failure data.

The failure strain is specified as a tabulated function of stress triaxiality and strain
rate: $\varepsilon_f = \varepsilon_f(\eta, \dot{\varepsilon})$. Bilinear interpolation
is used between data points.

---

#### Johnson-Cook Failure

**Parameters:** $D_1$ through $D_5$, $\dot{\varepsilon}_0$, $T_{\text{melt}}$, $T_{\text{room}}$

**Failure strain:**

$$
\varepsilon_f = \left[D_1 + D_2 \exp(D_3 \eta)\right]
\left[1 + D_4 \ln\frac{\dot{\varepsilon}}{\dot{\varepsilon}_0}\right]
\left[1 + D_5 T^*\right]
$$

where $\eta = p / \sigma_{\text{vm}}$ is the stress triaxiality. Damage accumulates as $D = \sum \Delta\varepsilon^p / \varepsilon_f$.

**Application:** Ballistic impact, penetration, metal fragmentation.

---

#### Cockcroft-Latham Failure

**Parameters:** $W_c$ (critical plastic work)

**Failure criterion:**

$$
W = \int_0^{\bar{\varepsilon}^p} \max(\sigma_1, 0) \, d\bar{\varepsilon}^p \geq W_c
$$

where $\sigma_1$ is the maximum principal stress. Failure occurs when the accumulated
plastic work under tension reaches the critical value.

**Application:** Metal cutting, machining, ductile fracture under tension.

---

#### Lemaitre CDM (Continuum Damage Mechanics)

**Parameters:** $S_0$ (damage strength), $s$ (damage exponent), $D_c$ (critical damage), $\varepsilon_D$ (damage threshold strain)

**Damage evolution:**

$$
\dot{D} = \left(\frac{Y}{S_0}\right)^s \dot{\bar{\varepsilon}}^p, \qquad
Y = \frac{\sigma_{\text{eq}}^2}{2E(1-D)^2}R_\nu
$$

where $R_\nu = \frac{2}{3}(1+\nu) + 3(1-2\nu)\eta^2$ is the triaxiality function.

**Application:** Low-cycle fatigue, coupled damage-plasticity analyses.

---

#### Puck Failure Criteria

**Parameters:** $R_\parallel^{(+)}$, $R_\parallel^{(-)}$ (fiber strengths), $R_\perp^{(+)}$, $R_\perp^{(-)}$ (matrix strengths), $R_{\perp\parallel}$, $p_{\perp\parallel}^{(+)}$, $p_{\perp\parallel}^{(-)}$ (inclination parameters)

The Puck model distinguishes fiber failure (direct stress criterion) from inter-fiber
failure (action-plane based) with separate modes for matrix tension, compression, and
shear.

**Application:** Laminated composites with physically-based matrix failure characterization.

---

#### FLD (Forming Limit Diagram) Failure

**Parameters:** Forming limit curve as pairs $(\varepsilon_2, \varepsilon_1^{\text{limit}})$

Failure is detected when the in-plane principal strains $(\varepsilon_2, \varepsilon_1)$
cross the user-supplied forming limit curve. Linear interpolation is used between
data points.

**Application:** Sheet metal stamping, formability assessment.

---

#### Wilkins Failure

**Parameters:** $D_c$ (critical damage), $a$, $\beta$ (pressure weight), $w$ (deviatoric weight)

**Damage accumulation:**

$$
D = \int \left(\frac{1}{1 - a \, P^*}\right)^{\beta}
\left(\frac{2 - A_d}{A_d}\right)^{w} d\bar{\varepsilon}^p
$$

where $P^*$ is the normalized pressure and $A_d$ is the asymmetry of the deviatoric
stress tensor.

**Application:** Metal fragmentation, expanding ring/cylinder problems.

---

#### Tuler-Butcher Failure

**Parameters:** $\sigma_c$ (threshold stress), $K$ (spall integral limit), $\lambda$ (exponent)

**Spall criterion:**

$$
\int_0^t (\sigma_1 - \sigma_c)^\lambda \, dt \geq K
$$

Failure occurs when the time-integrated excess tensile stress exceeds the material
spall constant $K$.

**Application:** Spallation under explosive loading, plate impact.

---

#### Maximum Stress Failure

**Parameters:** $X_T$, $X_C$, $Y_T$, $Y_C$, $Z_T$, $Z_C$, $S_{12}$, $S_{23}$, $S_{13}$ (component stress limits)

Failure occurs when any stress component exceeds its corresponding allowable:
$\sigma_{11} > X_T$ or $|\sigma_{11}| > X_C$, etc. Each mode is tracked independently.

**Application:** Brittle materials, composites with non-interacting failure modes.

---

#### Maximum Strain Failure

**Parameters:** $\varepsilon_1^T$, $\varepsilon_1^C$, $\varepsilon_2^T$, $\varepsilon_2^C$, $\varepsilon_3^T$, $\varepsilon_3^C$, $\gamma_{12}^f$, $\gamma_{23}^f$, $\gamma_{13}^f$

Failure occurs when any strain component exceeds its corresponding allowable limit.
Each mode is tracked independently, analogous to the maximum stress criterion but
evaluated in strain space.

**Application:** Composites and ceramics where strain limits are better characterized than stress limits.

---

#### Energy-Based Failure

**Parameters:** $W_c$ (critical strain energy density)

**Failure criterion:**

$$
W = \frac{1}{2} \sigma_{ij} \varepsilon_{ij} \geq W_c
$$

Failure occurs when the total strain energy density at an integration point reaches the
critical threshold.

**Application:** General-purpose failure criterion for materials with known energy absorption capacity.

---

#### Wierzbicki Failure (Modified Mohr-Coulomb)

**Parameters:** $c_1$, $c_2$, $c_{\theta}^s$, $c_{\theta}^{ax}$ (calibration constants)

**Failure strain:**

$$
\varepsilon_f = \left(\frac{c_{\theta}^{ax}}{c_{\theta}}\right)
\left[c_1 e^{-c_2 \eta} \right]
$$

where $c_\theta$ depends on the normalized Lode angle parameter $\bar{\theta}$, capturing
the effect of stress state beyond triaxiality alone.

**Application:** Sheet metal fracture under complex stress states (bending, shear, biaxial).

---

#### Fabric Failure

**Parameters:** $\varepsilon_1^f$ (warp failure strain), $\varepsilon_2^f$ (weft failure strain), $\gamma_{12}^f$ (shear failure strain)

Biaxial textile failure criterion with independent warp, weft, and shear failure modes.
Upon failure in one direction, the corresponding stress component is zeroed while the
remaining directions continue to carry load.

**Application:** Airbag tear, parachute failure, woven composite damage.

---

### Wave 19: Extended Failure Models

The following ten failure models are available in `physics/failure/failure_wave19.hpp`
(namespace `nxs::failure`):

#### LaDeveze Delamination

Meso-scale damage mechanics model for interlaminar failure. Separate damage
variables track mode I (opening) and mode II (shearing) delamination with
thermodynamic conjugate forces driving damage evolution. The model captures
the coupling between in-plane damage and delamination initiation.

**Application:** Composite delamination under impact and fatigue loading.

---

#### Hoffman Failure Criterion

A generalized quadratic interaction criterion for anisotropic materials that
distinguishes between tensile and compressive strengths (unlike the symmetric
Hill criterion):

$$
F_1\sigma_1 + F_2\sigma_2 + F_{11}\sigma_1^2 + F_{22}\sigma_2^2
+ F_{12}\sigma_1\sigma_2 + F_{66}\sigma_{12}^2 \geq 1
$$

**Application:** Composites and other anisotropic materials with asymmetric strength.

---

#### Tsai-Hill Failure Criterion

A simplified quadratic interaction criterion derived from Hill's anisotropic
yield criterion applied to composites:

$$
\left(\frac{\sigma_1}{X}\right)^2
- \frac{\sigma_1 \sigma_2}{X^2}
+ \left(\frac{\sigma_2}{Y}\right)^2
+ \left(\frac{\sigma_{12}}{S}\right)^2 \geq 1
$$

**Application:** Unidirectional composite laminates with moderate stress interaction.

---

#### RTCl (Rice-Tracey-Cockcroft-Latham)

Combined triaxiality and maximum-stress ductile failure criterion. Damage
accumulates as a weighted integral of plastic strain increments where the
weight depends on both stress triaxiality (Rice-Tracey) and maximum principal
stress (Cockcroft-Latham).

**Application:** Ductile metal fracture under complex stress states.

---

#### Mullins Damage

Stress-softening (Mullins effect) model for filled elastomers. Damage is
driven by the maximum historical strain energy density. Subsequent loading
below this threshold exhibits reduced stiffness, with recovery upon exceeding
the historical maximum.

**Application:** Filled rubber components, elastomeric bearings, tires.

---

#### Spalling Failure

Tensile pressure-based spalling criterion for materials under shock loading.
Failure occurs when the hydrostatic tension exceeds the spall strength $P_{\text{spall}}$.
The model supports both instantaneous spall and progressive damage accumulation.

**Application:** Plate impact, explosive loading, dynamic fragmentation.

---

#### HC_DSSE (Hosford-Coulomb with DSSE)

Combined Hosford-Coulomb fracture initiation with the Domain of Shell-to-Shell
Equivalence (DSSE) correction for proportional and non-proportional loading paths.
The failure locus depends on stress triaxiality, Lode angle parameter, and
equivalent plastic strain.

**Application:** Sheet metal forming, crashworthiness with complex loading paths.

---

#### Adhesive Joint Failure

Mixed-mode failure criterion for structural adhesive joints. The criterion
combines normal (peel) and shear stress components with a power-law interaction:

$$
\left(\frac{\sigma_n}{\sigma_n^c}\right)^a
+ \left(\frac{\tau}{\tau^c}\right)^b \geq 1
$$

**Application:** Bonded automotive joints, composite repair patches.

---

#### Windshield Failure

Multi-layer failure model for laminated glass windshields. Each glass ply
uses a Weibull-distributed tensile strength criterion, while the PVB interlayer
uses a strain-based criterion. Failed glass elements retain compressive
stiffness while losing tensile capacity.

**Application:** Automotive windshield impact, bird strike, pedestrian safety.

---

#### Generalized Energy Failure

Extended energy-based failure criterion with separate contributions from
tensile, compressive, and shear strain energy densities, each with independent
critical thresholds and weighting factors.

**Application:** General-purpose failure for materials with direction-dependent energy limits.

---

(ch11_eos)=
## Equations of State

Thirteen equation-of-state (EOS) models are provided in `physics/eos.hpp`. These are used
in conjunction with material models for problems involving high pressures, shock waves,
or compressible media.

### EOS Data Structure

```cpp
struct EOSState {
    Real pressure;
    Real density, density_ref;
    Real internal_energy;
    Real sound_speed;
    Real temperature;
    Real compression;        // μ = ρ/ρ₀ − 1
};

class EquationOfState {
public:
    virtual void compute_pressure(EOSState& state) = 0;
    virtual Real sound_speed(const EOSState& state) = 0;
};
```

### EOS Model Reference

#### Ideal Gas

$$
P = (\gamma - 1) \, \rho \, e
$$

where $\gamma$ is the ratio of specific heats and $e$ is the specific internal energy.

**Application:** Low-pressure gas dynamics.

---

#### Mie-Grüneisen

$$
P = \frac{\rho_0 c_0^2 \mu
\left[1 + \left(1 - \frac{\Gamma_0}{2}\right)\mu\right]}
{\left[1 - (S_1 - 1)\mu - S_2\frac{\mu^2}{\mu+1}
- S_3\frac{\mu^3}{(\mu+1)^2}\right]^2}
+ \Gamma_0 \rho e
$$

where $\mu = \rho/\rho_0 - 1$, $c_0$ is the bulk sound speed, $S_1, S_2, S_3$ are the
shock velocity–particle velocity coefficients, and $\Gamma_0$ is the Grüneisen parameter.

**Application:** Shock-loaded metals (copper, aluminum, steel).

---

#### JWL (Jones-Wilkins-Lee)

$$
P = A \, e^{-R_1 V/V_0} + B \, e^{-R_2 V/V_0}
+ \frac{\omega \, e}{V}
$$

where $V/V_0$ is the relative volume, and $A$, $B$, $R_1$, $R_2$, $\omega$ are material
parameters.

**Application:** Detonation products from high explosives.

---

#### Polynomial

$$
P = C_0 + C_1\mu + C_2\mu^2 + C_3\mu^3 + (C_4 + C_5\mu) \, e
$$

**Application:** General-purpose EOS for a wide range of materials.

---

#### Tabulated

$$
P = P(\rho, e) \quad \text{from lookup table}
$$

Bilinear interpolation on a regular grid of density and internal energy values.

**Application:** Materials with experimentally measured EOS data.

---

#### Murnaghan

**Parameters:** $K_0$ (bulk modulus), $K_0'$ (pressure derivative), $\rho_0$

$$
P = \frac{K_0}{K_0'}\left[\left(\frac{\rho}{\rho_0}\right)^{K_0'} - 1\right]
$$

**Application:** Third-order elastic description of solids at moderate pressures.

---

#### Noble-Abel

**Parameters:** $\gamma$ (ratio of specific heats), $b$ (covolume)

$$
P = \frac{(\gamma - 1) \rho \, e}{1 - b \, \rho}
$$

The covolume $b$ accounts for the finite molecular volume, modifying the ideal gas law
at high densities.

**Application:** Propellant gases, combustion products at high density.

---

#### Stiff Gas (Tait)

**Parameters:** $B$ (pressure constant), $\gamma$ (Tait exponent), $\rho_0$

$$
P = B\left[\left(\frac{\rho}{\rho_0}\right)^{\gamma} - 1\right]
$$

**Application:** Water and other weakly compressible liquids, underwater explosions.

---

#### Tillotson

**Parameters:** $a$, $b$, $A$, $B$, $E_0$, $\alpha$, $\beta$, $E_{\text{iv}}$, $E_{\text{cv}}$

The Tillotson EOS uses separate formulations for compressed, cold expanded, and hot
expanded states, with a smooth interpolation region between incipient and complete
vaporization energies ($E_{\text{iv}}$ and $E_{\text{cv}}$).

**Application:** Hypervelocity impact, planetary collision, meteorite impact simulations.

---

#### Sesame

**Parameters:** 2D lookup table $P(\rho, T)$, $e(\rho, T)$

Pressure and energy are obtained by bilinear interpolation from a 2D thermodynamic
table (typically from the SESAME database). Temperature and density serve as the
independent variables.

**Application:** Multi-phase materials, high-energy-density physics, weapon effects.

---

#### PowderBurn

**Parameters:** $\rho_s$ (solid density), $A_c$ (burn rate coefficient), $n$ (pressure exponent), EOS parameters for gas and solid phases

$$
\dot{F} = A_c \, P^n (1 - F)^{2/3}
$$

where $F$ is the burned mass fraction. The model couples a surface-regression burn rate
with separate EOS for the solid propellant and product gas phases.

**Application:** Propellant combustion, gun interior ballistics, reactive materials.

---

#### Compaction

**Parameters:** Loading curve $P(\rho)$, unloading bulk modulus $K_u$

The compaction EOS follows a nonlinear loading path defined by a tabulated
pressure-density curve. Unloading is elastic with a user-specified bulk modulus,
producing permanent densification.

**Application:** Porous metals, powder compacts, geological compaction.

---

#### Osborne

**Parameters:** $A_1$ through $A_6$ (polynomial coefficients)

$$
P = \frac{A_1 \mu + A_2 \mu^2 + A_3 \mu^3 + (A_4 + A_5 \mu + A_6 \mu^2) e}{e + 1}
$$

where $\mu = \rho / \rho_0 - 1$. This extended polynomial form provides stronger
coupling between compression and internal energy than the standard polynomial EOS.

**Application:** Metals and geomaterials requiring pressure-energy coupling.

---

(ch12_composites)=
## Composite Analysis

The composite analysis module (`physics/composite/`) implements Classical Lamination
Theory (CLT) with extensions for thermal residual stress, interlaminar shear stress, and
progressive failure analysis.

### Ply Definition

```cpp
struct Ply {
    Real thickness;
    Real angle;                // Fiber orientation (degrees)
    Real E1, E2, G12, nu12;   // Lamina elastic properties
    Real Xt, Xc, Yt, Yc, S;   // Strength properties
    Real alpha1, alpha2;       // Coefficients of thermal expansion
};
```

### ABD Matrix Computation

The `CompositeLaminate` class computes the ABD stiffness matrix from the laminate
definition:

$$
\begin{bmatrix} \mathbf{N} \\ \mathbf{M} \end{bmatrix}
= \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B} & \mathbf{D} \end{bmatrix}
\begin{bmatrix} \boldsymbol{\varepsilon}^0 \\ \boldsymbol{\kappa} \end{bmatrix}
$$

where:

$$
A_{ij} = \sum_{k=1}^{N} \bar{Q}_{ij}^{(k)} (z_k - z_{k-1})
$$

$$
B_{ij} = \frac{1}{2} \sum_{k=1}^{N} \bar{Q}_{ij}^{(k)} (z_k^2 - z_{k-1}^2)
$$

$$
D_{ij} = \frac{1}{3} \sum_{k=1}^{N} \bar{Q}_{ij}^{(k)} (z_k^3 - z_{k-1}^3)
$$

Here $\bar{Q}_{ij}^{(k)}$ is the transformed reduced stiffness matrix for ply $k$, and
$z_k$ is the distance from the laminate midplane to the top of ply $k$.

### Thermal Residual Stress

The `CompositeThermal` class computes thermal resultants and ply-level stresses due to a
temperature change $\Delta T$:

$$
\mathbf{N}^T = \sum_{k=1}^{N} \bar{\mathbf{Q}}^{(k)} \boldsymbol{\alpha}^{(k)} \Delta T
\, (z_k - z_{k-1})
$$

$$
\mathbf{M}^T = \frac{1}{2} \sum_{k=1}^{N} \bar{\mathbf{Q}}^{(k)} \boldsymbol{\alpha}^{(k)}
\Delta T \, (z_k^2 - z_{k-1}^2)
$$

### Interlaminar Shear Stress

The `CompositeInterlaminar` class computes through-thickness shear stresses using
equilibrium equations, which are critical for delamination assessment.

### Progressive Failure Analysis

The `CompositeProgressiveFailure` class implements:

1. **First-Ply Failure (FPF)** — Detection of the first ply to fail under applied loading
   using Hashin, Tsai-Wu, or maximum stress criteria.

2. **Ply degradation** — After failure, the stiffnesses of the failed ply are reduced:
   - **Selective degradation:** Only the moduli associated with the failure mode are set
     to near-zero (e.g., fiber failure zeros $E_1$; matrix failure zeros $E_2$ and $G_{12}$).
   - **Discount method:** All stiffnesses are set to near-zero.

3. **Strength envelope generation** — A polar plot of failure load as a function of
   loading angle, computed by evaluating the failure criterion at multiple load orientations.
