# NexusSim Software Manual

**Version** 1.0.0 | **Date** March 2026 | **Framework** C++20 / Kokkos / MPI | **License** Apache 2.0

---

## Document Information

| Field            | Value                                              |
|------------------|----------------------------------------------------|
| Document Title   | NexusSim Software Manual                           |
| Version          | 1.0.0                                              |
| Date             | March 2026                                         |
| Classification   | Unclassified / Public                              |
| Copyright        | 2024–2026 NexusSim Development Team                |
| License          | Apache License, Version 2.0                        |

## Scope and Audience

This manual serves as the authoritative reference for the NexusSim computational mechanics
framework. It is intended for the following audiences:

- **Simulation engineers** who use NexusSim to set up and execute analyses.
- **Developers** who extend the framework with new elements, materials, or solvers.
- **Researchers** who require detailed knowledge of the underlying mathematical formulations.

The document assumes familiarity with continuum mechanics, the finite element method, and
C++ programming.

## Typographic Conventions

Throughout this manual, the following conventions are observed:

| Convention                  | Meaning                                          |
|-----------------------------|--------------------------------------------------|
| `monospace`                 | Source code, class names, file names, commands    |
| *italic*                    | Mathematical variables, emphasis                 |
| **bold**                    | Key terms upon first introduction                |
| $\sigma_{ij}$              | Mathematical notation (LaTeX)                    |
| "See {ref}`Section 5.1 <ch05_infrastructure>`" | Cross-reference to another section |

## Symbol Conventions

| Symbol                | Meaning                                              |
|-----------------------|------------------------------------------------------|
| $\boldsymbol{\sigma}$ | Cauchy stress tensor                                |
| $\boldsymbol{\varepsilon}$ | Infinitesimal strain tensor                    |
| $\mathbf{F}$          | Deformation gradient                                |
| $\mathbf{B}$          | Left Cauchy–Green deformation tensor ($\mathbf{F}\mathbf{F}^T$) |
| $J$                   | Jacobian determinant ($\det\mathbf{F}$)             |
| $E$, $\nu$            | Young's modulus, Poisson's ratio                    |
| $\lambda$, $\mu$      | Lamé parameters                                     |
| $\rho$                | Mass density                                        |
| $\delta$              | Peridynamic horizon                                 |
| $h$                   | SPH smoothing length                                |
| $\mathbf{K}$          | Stiffness matrix                                    |
| $\mathbf{M}$          | Mass matrix                                         |
| $\mathbf{u}$          | Displacement vector                                 |
| $\mathbf{v}$          | Velocity vector                                     |
| $\Delta t$            | Time step                                           |
| $N_i$                 | Shape function for node $i$                         |

---

```{toctree}
:maxdepth: 3
:numbered:
:caption: Contents

user_guide
core_framework
elements
constitutive
solvers
contact_constraints
multiphysics
advanced
verification
phase6
material_catalog
migration_guide
appendices
```
