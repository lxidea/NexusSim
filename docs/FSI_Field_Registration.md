# FSI Prototype – Field Registration & DualView Setup
> See `README_NEXTGEN.md` for documentation map and reading order.

## Goal

Deliver Step 1 from the FSI prototype plan by defining concrete FieldRegistry entries and associated Kokkos `DualView` layouts for structural displacement, fluid traction, and mesh motion fields.[FSI_Prototype_Plan.md:9]

## Field Definitions

| Field Name | Domain | Location | Components | Data Layout | Notes |
| --- | --- | --- | --- | --- | --- |
| `solid.displacement` | Structural FEM | Nodal | 3 (x, y, z) | `Kokkos::DualView<double*[3], Kokkos::LayoutRight>` | Primary solver state; updated each structural time step; host mirror used for legacy validation. |
| `solid.velocity` | Structural FEM | Nodal | 3 | `DualView<double*[3]>` | Required for explicit integrator and coupling diagnostics. |
| `fluid.traction` | Fluid ALE | Surface nodes on interface | 3 | `DualView<double*[3]>` | Output from fluid solver; consumed by `traction_to_force` operator. |
| `fluid.mesh_motion` | Fluid ALE | Nodal | 3 | `DualView<double*[3]>` | Target field for displacement mapping; drives mesh smoothing kernel. |
| `coupling.interface_mask` | Shared | Nodal (boolean) | 1 | `DualView<int*>` | Identifies interface nodes; maintained by `InterfaceTracker`. |

## Registration API (C++20 Sketch)

```cpp
struct FieldDescriptor {
    std::string name;
    FieldLocation location;
    FieldType type;
    size_t components;
    Kokkos::DualView<double*> base_view;  // Overloaded per component count
};

class FieldRegistry {
public:
    template<typename ViewType>
    void register_field(std::string_view name,
                        FieldLocation location,
                        FieldType type,
                        ViewType view) {
        fields_.emplace(name, FieldEntry{location, type, view});
    }

    template<typename ViewType>
    ViewType get_view(std::string_view name);
};
```

## Initialization Steps

1. **Allocate DualViews**  
   - Structural solver allocates `solid.displacement` and `solid.velocity` using current nodal count (`num_solid_nodes`).  
   - Fluid solver allocates `fluid.traction` and `fluid.mesh_motion` using interface mesh size (`num_fluid_nodes`).  
   - `coupling.interface_mask` created by mesh services during preprocessing (Wave 1 deliverable). [Legacy_Migration_Roadmap.md:18]

2. **Register with FieldRegistry**  
   - During solver setup, each module calls `FieldRegistry::register_field` with associated DualView and metadata.  
   - CouplingManager obtains handles by name to bind CouplingOperators.

3. **Mirror Synchronisation**  
   - After structural solve: `solid.displacement.modify_device()`, execute solver updates, then `sync_host()` if CPU consumers (logging) need data.  
   - After fluid solve: similar pattern for `fluid.traction`.

4. **Interface Mask Maintenance**  
   - InterfaceTracker updates mask on device; propagate to host only when diagnostics or CLI tools request. [Coupling_GPU_Specification.md:22]

## Deliverables

- Header stub (`framework/coupling/FieldRegistry.hpp`) containing descriptor types and registration API.
- Unit test scaffold verifying registration and retrieval for each field (Catch2 test case `TEST_CASE("fsi_field_registry_dualview")`).
- Updated `FSI_Prototype_Plan.md` checklist marking Step 1 as in progress.
