# NexusSim YAML Input Format Specification

**Version**: 0.1.0
**Date**: 2025-11-05
**Status**: Draft

## Overview

NexusSim uses YAML-based configuration files for simulation setup. This document specifies the complete format, including all supported sections, keywords, and data types.

---

## File Structure

A NexusSim input file consists of the following top-level sections:

```yaml
simulation:       # Solver type, time integration, damping
mesh:             # Mesh file and element blocks
materials:        # Material property definitions
boundary_conditions:  # BCs (displacement, force, gravity)
initial_conditions:   # Initial velocities, displacements (optional)
output:           # Output format, frequency, fields
advanced:         # GPU, MPI, element settings (optional)
```

All sections use **2-space indentation** (YAML standard).

---

## 1. `simulation` Section

Controls solver type, time integration, and damping.

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Simulation name | `"cantilever_beam"` |
| `type` | string | Solver type | `"explicit_dynamics"` |
| `final_time` | float | End time (seconds) | `0.01` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | `""` | User description |
| `time_step` | float | `0.0` | Timestep (0=auto CFL) |
| `cfl_factor` | float | `0.9` | CFL safety factor |

### Damping Subsection (Optional)

```yaml
simulation:
  damping:
    type: "rayleigh"    # rayleigh, artificial, none
    alpha: 0.0          # Mass-proportional
    beta: 0.001         # Stiffness-proportional
```

### Supported Solver Types

- `explicit_dynamics` - Central difference time integration
- `implicit_static` - Newton-Raphson (future)
- `modal_analysis` - Eigenvalue analysis (future)

---

## 2. `mesh` Section

Specifies mesh file and element block properties.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Path to mesh file (`.mesh`, `.exo`, `.k`) |

### Element Blocks

```yaml
mesh:
  file: "model.mesh"
  element_blocks:
    - name: "part_1"
      material: "aluminum"
      element_type: "HEX8"

    - name: "shell_part"
      material: "steel"
      element_type: "SHELL4"
      thickness: 0.001    # Required for shells (meters)
```

### Supported Element Types

| Type | Description | Nodes | Integration |
|------|-------------|-------|-------------|
| `HEX8` | 8-node hexahedron | 8 | 1-point (reduced) |
| `HEX20` | 20-node hexahedron | 20 | 27-point (3×3×3) |
| `TET4` | 4-node tetrahedron | 4 | 1-point |
| `TET10` | 10-node tetrahedron | 10 | 4-point |
| `SHELL4` | 4-node shell | 4 | 2×2 Gauss |
| `BEAM2` | 2-node beam | 2 | 2-point |
| `WEDGE6` | 6-node wedge/prism | 6 | 6-point (2×3) |

---

## 3. `materials` Section

Material property definitions.

### Linear Elastic Material

```yaml
materials:
  aluminum:
    type: "elastic"
    density: 2700.0    # kg/m³
    E: 69.0e9          # Young's modulus (Pa)
    nu: 0.33           # Poisson's ratio
```

### Future Material Models

```yaml
materials:
  steel_plastic:
    type: "elastic_plastic"
    density: 7850.0
    elastic:
      E: 210.0e9
      nu: 0.30
    plastic:
      model: "von_mises"
      yield_stress: 250.0e6
      hardening_modulus: 1.0e9
```

---

## 4. `boundary_conditions` Section

### Displacement BCs

```yaml
boundary_conditions:
  displacement:
    - nodeset: "fixed_nodes"
      dof: [0, 1, 2]      # 0=x, 1=y, 2=z
      value: 0.0

    - nodeset: "symmetry"
      dof: [0]            # Constrain only X
      value: 0.0
```

### Force BCs

```yaml
boundary_conditions:
  force:
    - nodeset: "loaded_nodes"
      dof: 2              # Z-direction
      value: -1000.0      # Newtons

    - nodeset: "distributed"
      dof: [0, 1, 2]
      value: [10.0, 0.0, -50.0]  # Force vector
```

### Gravity Load

```yaml
boundary_conditions:
  gravity:
    enabled: true
    acceleration: [0.0, 0.0, -9.81]  # m/s² (Earth gravity)
```

### DOF Numbering

- **0** = X-direction (translation)
- **1** = Y-direction (translation)
- **2** = Z-direction (translation)
- **3** = RX (rotation about X, for beams/shells)
- **4** = RY (rotation about Y)
- **5** = RZ (rotation about Z)

---

## 5. `initial_conditions` Section (Optional)

### Initial Velocity

```yaml
initial_conditions:
  velocity:
    - nodeset: "impactor"
      value: [0.0, 0.0, -5.0]  # 5 m/s downward
```

### Initial Displacement

```yaml
initial_conditions:
  displacement:
    - nodeset: "preload"
      value: [0.0, 0.0, 0.001]  # 1 mm prestress
```

---

## 6. `output` Section

### Visualization Output

```yaml
output:
  format: "vtk"           # vtk, exodus, hdf5
  directory: "./results"
  base_name: "simulation"
  frequency: 10           # Every 10 steps

  # OR time-based output:
  # time_interval: 0.001  # Every 1 ms
```

### Output Fields

```yaml
output:
  fields:
    nodal:
      - "displacement"
      - "velocity"
      - "acceleration"
    element:
      - "stress"
      - "strain"
      - "energy_density"
      - "plastic_strain"  # If plastic material
```

### Supported Formats

- `vtk` - ParaView VTK format (ASCII, `.vtk`)
- `exodus` - Exodus II (HDF5-based, `.exo`) - *future*
- `hdf5` - Raw HDF5 (`.h5`) - *future*

---

## 7. `advanced` Section (Optional)

### GPU Acceleration

```yaml
advanced:
  gpu:
    enabled: true
    device: 0    # CUDA device ID
```

### MPI Parallelization

```yaml
advanced:
  mpi:
    enabled: true
    partitioning: "metis"  # metis, parmetis, manual
```

### Element-Specific Settings

```yaml
advanced:
  elements:
    hex8:
      integration: "reduced"     # reduced (1-pt) or full (8-pt)
      hourglass_control: false   # Enable hourglass stabilization

    shell4:
      formulation: "belytschko_tsay"
      num_integration_points: 5  # Through-thickness
```

### Termination Criteria

```yaml
advanced:
  termination:
    max_time: 0.01
    max_steps: 1000000
    energy_ratio: 10.0   # Kinetic/Internal energy ratio
```

---

## Data Types

### Scalars

```yaml
time_step: 1.0e-4        # Float
max_steps: 1000          # Integer
enabled: true            # Boolean
name: "simulation"       # String
```

### Arrays

```yaml
dof: [0, 1, 2]           # Integer array
value: [1.0, 2.0, 3.0]   # Float array
```

### Scientific Notation

```yaml
E: 210.0e9               # 210 × 10⁹ Pa
E: 2.1e11                # Same value
time_step: 1.0e-6        # 1 microsecond
```

---

## Units Convention

NexusSim uses **SI units** throughout:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | meter | m |
| Mass | kilogram | kg |
| Time | second | s |
| Force | newton | N |
| Stress | pascal | Pa |
| Energy | joule | J |
| Density | kg/m³ | kg/m³ |
| Velocity | m/s | m/s |
| Acceleration | m/s² | m/s² |

**Derived Units**:
- Pressure/Stress: 1 Pa = 1 N/m²
- Young's Modulus: 1 GPa = 1×10⁹ Pa
- Common values:
  - Steel E: ~210 GPa = 210×10⁹ Pa = `210.0e9`
  - Aluminum E: ~70 GPa = `70.0e9`
  - Gravity: 9.81 m/s² = `9.81`

---

## Node Sets and Element Sets

Node sets and element sets are defined in the **mesh file** (`.mesh` format):

```
NODESETS fixed_end 4
0 3 6 9

NODESETS loaded_end 4
2 5 8 11

ELEMENTSETS beam_elements 2
0 1
```

These sets are referenced by name in the YAML file.

---

## Comments

```yaml
# This is a comment (starts with #)

simulation:
  name: "test"  # Inline comment

# Multi-line comment:
# This section defines materials
# Use standard SI units
materials:
  steel:
    density: 7850.0
```

---

## Complete Example

See `/examples/configs/comprehensive_example.yaml` for a fully annotated example demonstrating all features.

---

## Validation

The parser performs basic validation:
- Required fields must be present
- Numeric values must be parseable
- Nodesets/elementsets must exist in mesh file
- Material references must be defined
- Element types must be supported

For detailed error messages, run with logging level `INFO` or `DEBUG`.

---

## Future Extensions

Planned features for future versions:

- Load curves (time-dependent loads)
- Contact mechanics definitions
- Implicit solver parameters
- Modal analysis configuration
- Thermal coupling
- Multi-material interfaces
- Advanced material models (plasticity, damage, failure)

---

## See Also

- [Simple Mesh Format Specification](Simple_Mesh_Format.md)
- [LS-DYNA Keyword Compatibility](LSDYNA_Import.md) - *coming soon*
- [Material Library](Material_Library.md) - *coming soon*
- [Examples Directory](../examples/configs/)

---

**Document History**:
- v0.1.0 (2025-11-05): Initial specification
